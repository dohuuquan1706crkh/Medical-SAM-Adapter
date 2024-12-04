
import argparse
import os
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
#from dataset import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from loss import RecLoss
import cfg
import models.sam.utils.transforms as samtrans
import pytorch_ssim
#from models.discriminatorlayer import discriminator
from conf import settings
from utils import *
mae = nn.L1Loss(reduction="none")
#import matplotlib.pyplot as plt

# from lucent.modelzoo.util import get_model_layers
# from lucent.optvis import render, param, transform, objectives
# from lucent.modelzoo import inceptionv1

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
seed = torch.randint(1,11,(args.b,7))

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

def train_sam(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, schedulers=None, vis = 50):
    hard = 0
    epoch_loss = 0
    ind = 0
    # train mode
    net.train()
    optimizer.zero_grad()

    epoch_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    if args.loss == "DiceCELoss":
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        print("use DiceCELoss")
    elif args.loss == "BCEWithLogitsLoss":
        lossfunc = criterion_G
        print("use BCEWithLogitsLoss")
    elif args.loss == "evidential":
        lossfunc = RecLoss()
        print("use evidential")

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            # torch.cuda.empty_cache()
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            #print(imgs.shape)
            masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            #print(masks.shape)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            if 'pt' not in pack:
                imgs, pt, masks = generate_click_prompt(imgs, masks)
            else:
                pt = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']

            if args.thd:
                imgs, pt, masks = generate_click_prompt(imgs, masks)

                pt = rearrange(pt, 'b n d -> (b d) n')
                imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                masks = rearrange(masks, 'b c h w d -> (b d) c h w ')

                imgs = imgs.repeat(1,3,1,1)
                point_labels = torch.ones(imgs.size(0))

                imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
            showp = pt

            mask_type = torch.float32
            ind += 1
            b_size,c,w,h = imgs.size()
            longsize = w if w >=h else h

            if point_labels.clone().flatten()[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                if(len(point_labels.shape)==1): # only one point prompt
                    coords_torch, labels_torch, showp = coords_torch[None, :, :], labels_torch[None, :], showp[None, :, :]
                pt = (coords_torch, labels_torch)

            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
                #true_mask_ave = cons_tensor(true_mask_ave)
            # imgs = imgs.to(dtype = mask_type,device = GPUdevice)

            '''Train'''
            if args.mod == 'sam_adapt':
                if args.distributed != 'none':
                    for n, value in net.module.image_encoder.named_parameters():
                        if "Adapter" not in n:
                            value.requires_grad = False
                        else:
                            value.requires_grad = True
                else:
                    for n, value in net.image_encoder.named_parameters():
                        if "Adapter" not in n:
                            value.requires_grad = False
                        else:
                            value.requires_grad = True
            elif args.mod == 'sam_lora' or args.mod == 'sam_adalora':
                from models.common import loralib as lora
                lora.mark_only_lora_as_trainable(net.image_encoder)
                if args.mod == 'sam_adalora':
                    # Initialize the RankAllocator 
                    rankallocator = lora.RankAllocator(
                        net.image_encoder, lora_r=4, target_rank=8,
                        init_warmup=500, final_warmup=1500, mask_interval=10, 
                        total_step=3000, beta1=0.85, beta2=0.85, 
                    )
            else:
                for n, value in net.image_encoder.named_parameters(): 
                    value.requires_grad = True
            if args.distributed != 'none':
                imge, _ = net.module.image_encoder(imgs)   
            else:     
                imge, _ = net.image_encoder(imgs)
            with torch.no_grad():
                if args.net == 'sam' or args.net == 'mobile_sam':
                    se, de = net.module.prompt_encoder(points=pt, boxes=None, masks=None) if args.distributed != 'none' else net.prompt_encoder(points=pt, boxes=None, masks=None) 
                elif args.net == "efficient_sam":
                    coords_torch,labels_torch = transform_prompt(coords_torch,labels_torch,h,w)
                    se = net.prompt_encoder(
                        coords=coords_torch,
                        labels=labels_torch,
                    )
                    
            if args.net == 'sam':
                pred, _, _ = net.module.mask_decoder(
                    image_embeddings=imge, 
                    image_pe=net.module.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se, 
                    dense_prompt_embeddings=de, 
                    multimask_output=(args.multimask_output > 1)) if args.distributed != 'none' else net.mask_decoder(image_embeddings=imge, image_pe=net.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=se, dense_prompt_embeddings=de, multimask_output=(args.multimask_output > 1),) 
            elif args.net == 'mobile_sam':
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=(args.multimask_output > 1),
                )
            elif args.net == "efficient_sam":
                se = se.view(
                    se.shape[0],
                    1,
                    se.shape[1],
                    se.shape[2],
                )
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    multimask_output=(args.multimask_output > 1),
                )
                
            # Resize to the ordered output size
            pred = F.interpolate(pred,size=(args.out_size,args.out_size))
            #print(pred.shape, masks.shape)     [2, 1, 1024, 1024]   [2, 1, 1024, 1024]
            #print(pred.min(), pred.max(), masks.min(), masks.max()) 
            if args.loss == "evidential":
                loss = lossfunc(pred, masks, epoch)
                
                pbar.set_postfix(**{'loss (batch)': loss})
                # breakpoint()

                epoch_loss += loss.item()
            else: 
                loss = lossfunc(pred, masks)
                # breakpoint()

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                epoch_loss += loss.item()

            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            if args.mod == 'sam_adalora':
                (loss+lora.compute_orth_regu(net, regu_weight=0.1)).backward()
                optimizer.step()
                rankallocator.update_and_mask(net, ind)
            else:
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()

            '''vis images'''
            if vis:
                if ind % vis == 0:
                    namecat = 'Train'
                    for na in name[:2]:
                        namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
                    vis_image(imgs,pred,masks, None, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)

            pbar.update()

    return loss

def validation_sam(args, val_loader, epoch, net, clean_dir=True, val_mode='normal'):
    if val_mode == 'mc_dropout':
        net.eval()  # Ensure the model is in eval mode
        for module in net.modules():
            if isinstance(module, torch.nn.Dropout):
                print(f"Enabling dropout for {module}")
                module.train()  # Enable dropout

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,)*args.multimask_output*2
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    pred_ls = []
    mask_ls = []
    pred_var_ls = []
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            # breakpoint()
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            if 'pt' not in pack or args.thd:
                imgsw, ptw, masksw = generate_click_prompt(imgsw, masksw)
            else:
                ptw = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']
            
            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
                if args.thd:
                    pt = ptw[:,:,buoy: buoy + evl_ch]
                else:
                    pt = ptw

                imgs = imgsw[...,buoy:buoy + evl_ch]
                masks = masksw[...,buoy:buoy + evl_ch]
                buoy += evl_ch

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1,3,1,1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
                
                showp = pt

                mask_type = torch.float32
                ind += 1
                b_size,c,w,h = imgs.size()
                longsize = w if w >=h else h

                if point_labels.clone().flatten()[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    if(len(point_labels.shape)==1): # only one point prompt
                        coords_torch, labels_torch, showp = coords_torch[None, :, :], labels_torch[None, :], showp[None, :, :]
                    pt = (coords_torch, labels_torch)

                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    #true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                
                '''test'''
                with torch.no_grad():
                    if val_mode != 'deep_ensemble':
                        if args.distributed != 'none':
                            imge, encoder_attns = net.module.image_encoder(imgs)
                        else: 
                            imge, encoder_attns = net.image_encoder(imgs)
                        if args.net == 'sam' or args.net == 'mobile_sam':
                            se, de = net.module.prompt_encoder(points=pt, boxes=None, masks=None) if args.distributed != 'none' else net.prompt_encoder(points=pt, boxes=None, masks=None) 
                        elif args.net == "efficient_sam":
                            coords_torch,labels_torch = transform_prompt(coords_torch,labels_torch,h,w)
                            se = net.prompt_encoder(
                                coords=coords_torch,
                                labels=labels_torch,
                            )

                    if args.net == 'sam':
                        if val_mode == 'mc_dropout':
                            preds = []
                            for _ in range(10):
                                imge_i = F.dropout(imge, p=0.3, training=True)
                                pe_i = F.dropout(net.prompt_encoder.get_dense_pe(), p=0.3, training=True)
                                se_i = F.dropout(se, p=0.3, training=True)
                                de_i = F.dropout(de, p=0.3, training=True)
                                pred, _, decoder_attns = net.mask_decoder(
                                    image_embeddings=imge_i,
                                    image_pe=pe_i, 
                                    sparse_prompt_embeddings=se_i,
                                    dense_prompt_embeddings=de_i, 
                                    multimask_output=(args.multimask_output > 1),
                                )
                                preds.append(pred)
                            preds = torch.stack(preds, dim=0)
                            pred = preds.mean(dim=0)
                            pred_var = preds.var(dim=0)
                        elif val_mode == 'deep_ensemble':
                            preds = []
                            for net_i in net:
                                if args.distributed != 'none':
                                    imge, encoder_attns = net_i.module.image_encoder(imgs)
                                else: 
                                    imge, encoder_attns = net_i.image_encoder(imgs)
                                if args.net == 'sam' or args.net == 'mobile_sam':
                                    se, de = net_i.module.prompt_encoder(points=pt, boxes=None, masks=None) if args.distributed != 'none' else net_i.prompt_encoder(points=pt, boxes=None, masks=None) 
                                elif args.net == "efficient_sam":
                                    coords_torch,labels_torch = transform_prompt(coords_torch,labels_torch,h,w)
                                    se = net_i.prompt_encoder(
                                        coords=coords_torch,
                                        labels=labels_torch,
                                    )
                                pred, _, decoder_attns = net_i.mask_decoder(
                                        image_embeddings=imge,
                                        image_pe=net_i.prompt_encoder.get_dense_pe(), 
                                        sparse_prompt_embeddings=se,
                                        dense_prompt_embeddings=de, 
                                        multimask_output=(args.multimask_output > 1),
                                    )
                                preds.append(pred)
                            preds = torch.stack(preds, dim=0)
                            pred = preds.mean(dim=0)
                            pred_var = preds.var(dim=0)
                        else:
                            pred, _, decoder_attns = net.mask_decoder(
                                image_embeddings=imge,
                                image_pe=net.prompt_encoder.get_dense_pe(), 
                                sparse_prompt_embeddings=se,
                                dense_prompt_embeddings=de, 
                                multimask_output=(args.multimask_output > 1),
                            )
                    elif args.net == 'mobile_sam':
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de, 
                            multimask_output=(args.multimask_output > 1),
                        )
                    elif args.net == "efficient_sam":
                        se = se.view(
                            se.shape[0],
                            1,
                            se.shape[1],
                            se.shape[2],
                        )
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=se,
                            multimask_output=(args.multimask_output > 1),
                        )

                    # Resize to the ordered output size
                    #exitbreakpoint()
                    pred = F.interpolate(pred,size=(args.out_size,args.out_size))
                    if val_mode in ['mc_dropout', 'deep_ensemble']:
                        pred_var = F.interpolate(pred_var, size=(args.out_size, args.out_size))
                        pred_ls.append(pred)
                        mask_ls.append(masks)
                        pred_var_ls.append(pred_var)

                    tot += lossfunc(pred, masks).item()

                    '''vis images'''
                    if args.vis and ind % args.vis == 1:
                        # compute entropy map
                        #print("vis image")
                        x = torch.sigmoid(pred)
                        #x = x.view(pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3])
                        #print(sigmoid.shape)
                        x = -x*torch.log(x + 1e-8) - (1 - x) * torch.log(1 - x + 1e-8)
                        x = (x - x.amin(dim=(-1, -2), keepdim=True)) / (x.amax(dim=(-1, -2), keepdim=True) - x.amin(dim=(-1, -2), keepdim=True))
                        #print(entropy.shape)
                        x_ = mae(torch.sigmoid(pred), masks)
                        x_ = (x_ - x_.amin(dim=(-1, -2), keepdim=True)) / (x_.amax(dim=(-1, -2), keepdim=True) - x_.amin(dim=(-1, -2), keepdim=True))
                        #print(x_.shape)
                        namecat = 'Test'
                        for na in name[:2]:
                            img_name = na.split('/')[-1].split('.')[0]
                            namecat = namecat + img_name + '+'
                        # #print("encoder attn map")
                        # compose = [F.interpolate(imgs,size=(64,64)).detach().cpu().expand(imgs.shape[0], 3, 64, 64), F.interpolate(masks,size=(64,64)).detach().cpu().expand(masks.shape[0], 3, 64, 64)]
                        # for i, attn in enumerate(encoder_attns):
                        #   print(attn.shape)  
                        #   #attn = F.interpolate(attn, size=(128, 128), mode=)
                        #   #attn = (attn - attn.amin(dim=(-1, -2), keepdim=True)) / (attn.amax(dim=(-1, -2), keepdim=True) - attn.amin(dim=(-1, -2), keepdim=True)
                        #   attn = (attn - attn.amin(dim=(-1, -2), keepdim=True)) / (attn.amax(dim=(-1, -2), keepdim=True) - attn.amin(dim=(-1, -2), keepdim=True))
                        #   grad_attn = compute_gradient_map(attn)
                        #   #attn = attn.expand(attn.shape[0],3,attn.shape[1],attn.shape[2])
                        #   compose += [attn.expand(attn.shape[0],3,attn.shape[2],attn.shape[3])]
                        #   compose += [grad_attn.expand(attn.shape[0],3,attn.shape[2],attn.shape[3])]
                        #   # print(f"layer {i}: {attn.shape}")
                        # compose = torch.cat(compose, 0)
                        # vutils.save_image(compose, fp = os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + "_encoder_attn.jpg"), nrow = pred.shape[0], padding = 10)
                        # # print("decoder map")
                        # compose = [F.interpolate(imgs,size=(64,64)).detach().cpu().expand(imgs.shape[0], 3, 64, 64), F.interpolate(masks,size=(64,64)).detach().cpu().expand(masks.shape[0], 3, 64, 64)]
                        # for attn in decoder_attns:
                        #     attn = attn[1].mean(dim=-1).view(attn[1].shape[0], 64, 64).detach().cpu().unsqueeze(1)
                        #     attn = (attn - attn.amin(dim=(-1, -2), keepdim=True)) / (attn.amax(dim=(-1, -2), keepdim=True) - attn.amin(dim=(-1, -2), keepdim=True))
                        #     grad_attn = compute_gradient_map(attn)
                        #     compose += [attn.expand(attn.shape[0], 3, 64, 64)]
                        #     compose += [grad_attn.expand(attn.shape[0],3,attn.shape[2],attn.shape[3])]
                        # compose = torch.cat(compose, 0)
                        # vutils.save_image(compose, fp = os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + "_decoder_attn.jpg"), nrow = pred.shape[0], padding = 10)
                        # pred_sigmoid = torch.sigmoid(pred)
                        # error_map = torch.abs(masks - pred_sigmoid)
                        vis_image(imgs,pred, masks, x, x_, save_path=os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)
                        vis_image(imgs,pred_var, masks, x, x_, save_path=os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '_var.jpg'), reverse=False, points=showp)
                    # breakpoint()

                    temp = eval_seg(pred, masks, threshold)
                    mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            pbar.update()

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    if val_mode in ['mc_dropout', 'deep_ensemble']:
        # calculate correlation between predictions errors and uncertainty
        pred_ls = torch.cat(pred_ls, dim=0)
        pred_ls = (torch.sigmoid(pred_ls) > 0.5).float().squeeze(1)
        mask_ls = torch.cat(mask_ls, dim=0).squeeze(1)
        loss = (pred_ls != mask_ls).float().flatten(start_dim=1)
        # pred_ls = torch.cat([1 - pred_ls, pred_ls], dim=1)
        # mask_ls = torch.cat([1 - mask_ls, mask_ls], dim=1)
        # lss = nn.CrossEntropyLoss(reduction="none")
        # loss = lss(pred_ls, mask_ls) # consider this as the error

        pred_var_ls = torch.cat(pred_var_ls, dim=0).squeeze(1).flatten(start_dim=1)
        cov = (loss - loss.mean(axis=1, keepdims=True)) * (pred_var_ls - pred_var_ls.mean(axis=1, keepdims=True))
        pearson_corr = cov.mean(axis=1) / (loss.std(axis=1, unbiased=False) * pred_var_ls.std(axis=1, unbiased=False) + 1e-8)
        print(f"Average Pearson correlation: {pearson_corr.mean()}")

    return tot/ n_val , tuple([a/n_val for a in mix_res])

def transform_prompt(coord,label,h,w):
    coord = coord.transpose(0,1)
    label = label.transpose(0,1)

    coord = coord.unsqueeze(1)
    label = label.unsqueeze(1)

    batch_size, max_num_queries, num_pts, _ = coord.shape
    num_pts = coord.shape[2]
    rescaled_batched_points = get_rescaled_pts(coord, h, w)

    decoder_max_num_input_points = 6
    if num_pts > decoder_max_num_input_points:
        rescaled_batched_points = rescaled_batched_points[
            :, :, : decoder_max_num_input_points, :
        ]
        label = label[
            :, :, : decoder_max_num_input_points
        ]
    elif num_pts < decoder_max_num_input_points:
        rescaled_batched_points = F.pad(
            rescaled_batched_points,
            (0, 0, 0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
        label = F.pad(
            label,
            (0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
    
    rescaled_batched_points = rescaled_batched_points.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points, 2
    )
    label = label.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points
    )

    return rescaled_batched_points,label


def get_rescaled_pts(batched_points: torch.Tensor, input_h: int, input_w: int):
        return torch.stack(
            [
                torch.where(
                    batched_points[..., 0] >= 0,
                    batched_points[..., 0] * 1024 / input_w,
                    -1.0,
                ),
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * 1024 / input_h,
                    -1.0,
                ),
            ],
            dim=-1,
        )
        
