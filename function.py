
import argparse
import os
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import datetime

import gc
import wandb
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
from loss import RecLoss, GenGaussLoss, PCCLoss
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
    hard = 1
    epoch_loss = 0
    ind = 0
    # train mode
    net.train()
    accumulated_loss = 0.0
    optimizer.zero_grad()
    lambda_u = 0.001
    # lambda_u = epoch / 100
    # lambda_u = 1 / 500
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
    
    loss_uncert1 = GenGaussLoss()
    loss_uncert2 = PCCLoss()
    NUM_ACCUMULATION_STEPS = 4
    example_counter = 0
    if args.encoder == 'bayescap_decoder':
        print("use bayes_cap decoder")
    if args.encoder == 'sure_decoder':
        print("use sure decoder")
    if args.encoder == 'fft_decoder':
        print("use fft decoder")
    if args.encoder == 'fno_decoder':
        print("use fno decoder")
    
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        # breakpoint()
        for idx, pack in enumerate(train_loader):
            # torch.cuda.empty_cache()
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            #print(imgs.shape)
            masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            # imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
            # masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)
            #print(masks.shape)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            # breakpoint()
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
            # breakpoint()
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
                masks = (masks > 0).float()
                #true_mask_ave = cons_tensor(true_mask_ave)
            # imgs = imgs.to(dtype = mask_type,device = GPUdevice)

            '''Train'''
            if args.mod == 'sam_adapt' or args.mod == 'sam_adapt_dense':
                if args.distributed != 'none':
                    for n, value in net.module.image_encoder.named_parameters():
                        # print(n)
                        if "Adapter" not in n:
                            value.requires_grad = False
                        else:
                            value.requires_grad = True
                else:
                    for n, value in net.image_encoder.named_parameters():
                        # print(n)
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
                if args.encoder == 'bayescap_decoder':
                    pred, pred_a, pred_b, _, _ = net.module.mask_decoder(
                        image_embeddings=imge, 
                        image_pe=net.module.prompt_encoder.get_dense_pe(), 
                        sparse_prompt_embeddings=se, 
                        dense_prompt_embeddings=de, 
                        multimask_output=(args.multimask_output > 1)) if args.distributed != 'none' else net.mask_decoder(image_embeddings=imge, image_pe=net.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=se, dense_prompt_embeddings=de, multimask_output=(args.multimask_output > 1),) 
                elif args.encoder in ['sure_decoder', 'fft_decoder', 'fno_decoder']:    
                    pred, pred_var, _, _ = net.module.mask_decoder(
                        image_embeddings=imge, 
                        image_pe=net.module.prompt_encoder.get_dense_pe(), 
                        sparse_prompt_embeddings=se, 
                        dense_prompt_embeddings=de, 
                        multimask_output=(args.multimask_output > 1)) if args.distributed != 'none' else net.mask_decoder(image_embeddings=imge, image_pe=net.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=se, dense_prompt_embeddings=de, multimask_output=(args.multimask_output > 1),) 
                else:    
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
            if args.encoder == 'bayescap_decoder':
                pred_a = F.interpolate(pred_a,size=(args.out_size,args.out_size))
                pred_b = F.interpolate(pred_b,size=(args.out_size,args.out_size))
            elif args.encoder in ['sure_decoder', 'fft_decoder', 'fno_decoder']:
                pred_var = F.interpolate(pred_var,size=(args.out_size,args.out_size)) 

            if args.loss == "evidential":
                loss = lossfunc(pred, masks, epoch)
                
                pbar.set_postfix(**{'loss (batch)': loss})
                # breakpoint()

                epoch_loss += loss.item()
            else: 
                loss = lossfunc(pred, masks)
                # print(masks.max())
                # breakpoint()
                if args.encoder == 'bayescap_decoder':
                    loss_u = loss_uncert1(pred, pred_a, pred_b, masks)
                    # import IPython; IPython.embed(); exit(1)
                    loss = loss + loss_u * 1e-3
                elif args.encoder in ['sure_decoder', 'fft_decoder', 'fno_decoder']:
                    loss_u = loss_uncert2(pred, pred_var, masks)
                    # import IPython; IPython.embed(); exit(1)
                    loss = loss + loss_u * lambda_u

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                epoch_loss += loss.item()
                accumulated_loss += loss.item()
            # breakpoint()
            example_counter += args.b
            
                
            # if ((idx + 1) % NUM_ACCUMULATION_STEPS == 0) or (idx + 1 == len(train_loader)) or idx ==0:
            #     if args.encoder in {'bayescap_decoder', "sure_decoder"}:
            #         wandb.log({"train/loss": accumulated_loss/NUM_ACCUMULATION_STEPS, "train/loss_u": loss_u}, step=example_counter)
            #     else:
            #         wandb.log({"train/loss": accumulated_loss/NUM_ACCUMULATION_STEPS}, step=example_counter)                


            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            if args.mod == 'sam_adalora':
                loss /= NUM_ACCUMULATION_STEPS
                (loss+lora.compute_orth_regu(net, regu_weight=0.1)).backward()
                if ((idx + 1) % NUM_ACCUMULATION_STEPS == 0) or (idx + 1 == len(train_loader)):
                    if args.encoder in {'bayescap_decoder', "sure_decoder", "fft_decoder", "fno_decoder"}:
                        wandb.log({"train/loss": accumulated_loss/NUM_ACCUMULATION_STEPS, "train/loss_u": loss_u}, step=example_counter)
                    else:
                        wandb.log({"train/loss": accumulated_loss/NUM_ACCUMULATION_STEPS}, step=example_counter) 
                    optimizer.step()
                    optimizer.zero_grad()

                rankallocator.update_and_mask(net, ind)
            else:
                loss /= NUM_ACCUMULATION_STEPS
                loss.backward()
                if ((idx + 1) % NUM_ACCUMULATION_STEPS == 0) or (idx + 1 == len(train_loader)):
                    if args.encoder in {'bayescap_decoder', "sure_decoder", "fft_decoder", "fno_decoder"}:
                        wandb.log({"train/loss": accumulated_loss/NUM_ACCUMULATION_STEPS, "train/loss_u": loss_u}, step=example_counter)
                    else:
                        wandb.log({"train/loss": accumulated_loss/NUM_ACCUMULATION_STEPS}, step=example_counter) 
                    optimizer.step()
                    optimizer.zero_grad()
                    wandb.log({"train/loss": loss.item()})                
            
            '''vis images'''
            if vis:
                if ind % vis == 0:
                    namecat = 'Train'
                    for na in name[:2]:
                        namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
                    vis_image(imgs,pred,masks, None, save_path = os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)

            pbar.update()
            # break

    return loss

@torch.no_grad()
def validation_sam(args, val_loader, epoch, net, clean_dir=True, val_mode=args.val_mode):
    if val_mode == 'mc_dropout':
        net.eval()  # Ensure the model is in eval mode
        for module in net.modules():
            if isinstance(module, torch.nn.Dropout):
                print(f"Enabling dropout for {module}")
                module.train()  # Enable dropout
    if args.encoder == 'bayescap_decoder':
        loss_uncert = GenGaussLoss()
    elif args.encoder in ['sure_decoder', "fft_decoder", "fno_decoder"]:
        loss_uncert = PCCLoss()
    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,)*args.multimask_output*2
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot = 0
    hard = 1
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G
    pred_var_ls_min = []
    pred_var_ls_max = []
    pred_ls = []
    pred_ls_a = []
    pred_ls_b = []
    mask_ls = []
    pred_var_ls = []
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            # breakpoint()
            imgsw = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masksw = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            imgsw = torchvision.transforms.Resize((args.image_size,args.image_size))(imgsw)
            masksw = torchvision.transforms.Resize((args.out_size,args.out_size))(masksw)

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
                    masks = (masks > 0).float()
                    #true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                # print(masks.max())
                # print(torch.unique(masks))
                # breakpoint()
                
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
                            preds = torch.sigmoid(preds)
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
                            preds = torch.sigmoid(preds)
                            pred_var = preds.var(dim=0)
                        elif val_mode == 'ttdac':
                            preds = []
                            color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                            for _ in range(10):
                                imgs_i = color_jitter(imgs)
                                if args.distributed != 'none':
                                    imge_i, encoder_attns = net.module.image_encoder(imgs_i)
                                else: 
                                    imge_i, encoder_attns = net.image_encoder(imgs_i)
                                # pe_i = F.dropout(net.prompt_encoder.get_dense_pe(), p=0.3, training=True)
                                # se_i = F.dropout(se, p=0.3, training=True)
                                # de_i = F.dropout(de, p=0.3, training=True)
                                pred, _, decoder_attns = net.mask_decoder(
                                    image_embeddings=imge_i,
                                    image_pe=net.prompt_encoder.get_dense_pe(), 
                                    sparse_prompt_embeddings=se,
                                    dense_prompt_embeddings=de, 
                                    multimask_output=(args.multimask_output > 1),
                                )
                                preds.append(pred)
                            preds = torch.stack(preds, dim=0)
                            pred = preds.mean(dim=0)
                            preds = torch.sigmoid(preds)
                            pred_var = preds.var(dim=0)
                        elif val_mode == 'ttdap':
                            preds = []
                            for _ in range(10):
                                # imgs_i = color_jitter(imgs)
                                imge_i = imge + torch.randn_like(imge) * 0.1
                                # pe_i = F.dropout(net.prompt_encoder.get_dense_pe(), p=0.3, training=True)
                                # se_i = F.dropout(se, p=0.3, training=True)
                                # de_i = F.dropout(de, p=0.3, training=True)
                                pred, _, decoder_attns = net.mask_decoder(
                                    image_embeddings=imge_i,
                                    image_pe=net.prompt_encoder.get_dense_pe(), 
                                    sparse_prompt_embeddings=se,
                                    dense_prompt_embeddings=de, 
                                    multimask_output=(args.multimask_output > 1),
                                )
                                preds.append(pred)
                            preds = torch.stack(preds, dim=0)
                            pred = preds.mean(dim=0)
                            preds = torch.sigmoid(preds)
                            pred_var = preds.var(dim=0)
                        else:
                            if args.encoder == 'bayescap_decoder':
                                pred, pred_a, pred_b, _, _ = net.module.mask_decoder(
                                    image_embeddings=imge, 
                                    image_pe=net.module.prompt_encoder.get_dense_pe(), 
                                    sparse_prompt_embeddings=se, 
                                    dense_prompt_embeddings=de, 
                                    multimask_output=(args.multimask_output > 1)) if args.distributed != 'none' else net.mask_decoder(image_embeddings=imge, image_pe=net.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=se, dense_prompt_embeddings=de, multimask_output=(args.multimask_output > 1),) 
                            elif args.encoder in ['sure_decoder', 'fft_decoder', 'fno_decoder']:    
                                pred, pred_var, _, _ = net.module.mask_decoder(
                                    image_embeddings=imge, 
                                    image_pe=net.module.prompt_encoder.get_dense_pe(), 
                                    sparse_prompt_embeddings=se, 
                                    dense_prompt_embeddings=de, 
                                    multimask_output=(args.multimask_output > 1)) if args.distributed != 'none' else net.mask_decoder(image_embeddings=imge, image_pe=net.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=se, dense_prompt_embeddings=de, multimask_output=(args.multimask_output > 1),) 
                            else:    
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
                    #exitbreakpoint()
                    pred = F.interpolate(pred,size=(args.out_size,args.out_size))
                    if args.encoder == 'bayescap_decoder':
                        pred_a = F.interpolate(pred_a,size=(args.out_size,args.out_size)).clamp(min= 1e-4, max=1e3)
                        pred_b = F.interpolate(pred_b,size=(args.out_size,args.out_size)).clamp(min= 1e-4, max=1e3)
                        one_over_pred_a = 1 / pred_a
                        # uncertainty
                        # pred_var = (1 / pred_a**2) * torch.lgamma(3 / pred_b).exp() / torch.lgamma(1 / pred_b).exp()
                        pred_var = (one_over_pred_a**2) * torch.lgamma(3 / pred_b).exp().clamp(min= 1e-4, max=1e3) / torch.lgamma(1 / pred_b).exp().clamp(min= 1e-4, max=1e3)
                        # pred_var = (pred_a**2) * torch.lgamma(3 / pred_b) / torch.lgamma(1 / pred_b)
                        pred_ls_a.append(pred_a.cpu())
                        pred_ls_b.append(pred_b.cpu())

                        # loss_uncert = GenGaussLoss()
                        loss = loss_uncert(pred, pred_a, pred_b, masks)
                        # breakpoint()
                    elif args.encoder in ['sure_decoder', "fft_decoder", "fno_decoder"]:
                        pred, pred_var, _, _ = net.module.mask_decoder(
                            image_embeddings=imge, 
                            image_pe=net.module.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=se, 
                            dense_prompt_embeddings=de, 
                            multimask_output=(args.multimask_output > 1)) if args.distributed != 'none' else net.mask_decoder(image_embeddings=imge, image_pe=net.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=se, dense_prompt_embeddings=de, multimask_output=(args.multimask_output > 1),) 
                        # pred_ls.append(pred.cpu())
                        # mask_ls.append(masks.cpu())
                        # pred_var_ls.append(pred_var.cpu())
                        pred = F.interpolate(pred, size=(args.out_size, args.out_size))
                        pred_var = F.interpolate(pred_var, size=(args.out_size, args.out_size))
                        # loss_uncert = PCCLoss()
                        loss = loss_uncert(pred, pred_var, masks)
                        
                    if val_mode in ['mc_dropout', 'deep_ensemble', 'ttdac', 'ttdap', "SURE"]:
                        pred_var = F.interpolate(pred_var, size=(args.out_size, args.out_size))
                    if val_mode == "entropy":
                        pred_var = -torch.sigmoid(pred)*torch.log(torch.sigmoid(pred) + 1e-8) - (1 - torch.sigmoid(pred)) * torch.log(1 - torch.sigmoid(pred) + 1e-8)
                    pred_ls.append(pred.cpu())
                    mask_ls.append(masks.cpu())
                    pred_var_ls_min.append(pred_var.quantile(0.05).item())
                    pred_var_ls_max.append(pred_var.quantile(0.95).item())
                    pred_var_ls.append(pred_var.cpu())
                    # breakpoint()
                    tot += lossfunc(pred, masks).item()
                    temp = eval_seg(pred, masks, threshold)
                    '''vis images'''
                    # if args.vis and ind % args.vis == 0:
                    if args.vis and ind % args.vis == 0:
                    # if args.vis and ind % args.vis == 0 and temp[0] < 0.55:

                        vis_image_val(args, imgs, pred, masks, pred_var, name, epoch, reverse=False, points=showp)
                    # breakpoint()
                    mix_res = tuple([sum(a) for a in zip(mix_res, temp)])
                
            pbar.update()
            # break
    #####
    
    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)
    # breakpoint()
    
    if val_mode in ['mc_dropout', 'deep_ensemble', 'bayescap', 'ttdac', 'ttdap', "SURE", "entropy"]:
        # breakpoint()
        # calculate correlation between predictions errors and uncertainty
        if val_mode == "bayescap":
            pred_ls_a = torch.cat(pred_ls_a, dim=0).float().squeeze(1)
            pred_ls_b = torch.cat(pred_ls_b, dim=0).float().squeeze(1)   
        # pred_var_min = np.array(pred_var_ls_min).mean()
        # pred_var_max = np.array(pred_var_ls_max).mean()

        pred_ls = torch.cat(pred_ls, dim=0).float().squeeze(1)
        pred_logit = pred_ls
        pred_sigmoid = torch.sigmoid(pred_ls)
        pred_ls = (pred_sigmoid > 0.5)
        mask_ls = torch.cat(mask_ls, dim=0).squeeze(1)
        loss = (pred_ls != mask_ls).float()
        pred_var_ls = torch.cat(pred_var_ls, dim=0).squeeze(1)

        pearson_corr = calculate_pearson(loss, pred_var_ls)
        # print(f"Average Pearson correlation: {pearson_corr}")
        loss = loss.flatten(start_dim=0)
        pred_var_ls = pred_var_ls.flatten(start_dim=0)
        pred_var_min = pred_var_ls.min()
        pred_var_max = pred_var_ls.max()
        uce = calculate_uce(loss, pred_var_ls, pred_var_min, pred_var_max)
        # print(f"UCE: {uce}")
        # map = (loss>0.5)|(pred_sigmoid.flatten(start_dim=0)>0.5)
        # loss_map = loss[map]
        # pred_var_ls_map = pred_var_ls[map]  
        # pearson_corr_map = calculate_pearson(loss_map, pred_var_ls_map)
        # print(f"Average Pearson_correlation_map: {pearson_corr_map}")
        # uce_map = calculate_uce(loss_map, pred_var_ls_map, pred_var_min, pred_var_max)      
        # print(f"UCE_map: {uce_map}")
        # breakpoint()
        if args.plot_histogram:

            preds_prob = torch.where(pred_sigmoid > 0.5, pred_sigmoid, 1 - pred_sigmoid)
            correct_predictions = (1 - loss).bool()

            # Convert tensors to numpy arrays
            confidence_scores = preds_prob.flatten().cpu().numpy()
            correct_predictions = correct_predictions.flatten().cpu().numpy()

            plot_confidence_histogram(confidence_scores, correct_predictions, epoch, args)
            pred_logit = pred_logit.flatten().cpu().numpy()

            plot_logit_histogram(pred_logit, epoch, args)
            pred_var_ls = pred_var_ls.cpu().numpy()
            
            plot_var_histogram(pred_var_ls, epoch, args)
            
            if val_mode == "bayescap":
                pred_ls_a = pred_ls_a.flatten().cpu().numpy()
                plot_alpha_histogram(pred_ls_a, epoch, args)
                pred_ls_b = pred_ls_b.flatten().cpu().numpy()
                plot_beta_histogram(pred_ls_b, epoch, args)

                del pred_ls_a, pred_ls_b
                gc.collect()
                torch.cuda.empty_cache()
    
    del pred_ls, mask_ls, pred_var_ls
    gc.collect()
    torch.cuda.empty_cache()
    return tot/ n_val, pearson_corr, uce, tuple([a/n_val for a in mix_res])

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
        
def calculate_uce(error, uncertainties, min_uncertainty, max_uncertainty, num_bins=10, task_type='classification'):
    """
    Calculate the Uncertainty Calibration Error (UCE) for classification or regression tasks.
    Args:
        loss (torch.Tensor): Loss values associated with each prediction.
        uncertainties (torch.Tensor): Uncertainty values associated with each prediction (0 to 1 for classification; unbounded for regression).
        num_bins (int): Number of bins to divide the uncertainty range [0, 1].
        task_type (str): 'classification' or 'regression', indicating the type of task.
    Returns:
        float: Calculated UCE value.
    """
    # Normalize uncertainties to [0, 1]
    # min_uncertainty = uncertainties.min()
    # max_uncertainty = uncertainties.max()
    uncertainties = (uncertainties - min_uncertainty) / (max_uncertainty - min_uncertainty+1e-8)
    uncertainties[uncertainties>1] = 1
    uncertainties[uncertainties<0] = 0
    uncertainties = torch.clamp(uncertainties, 0, 1)

    bin_edges = torch.linspace(0, 1, num_bins + 1, device=uncertainties.device)
    bin_indices = torch.bucketize(uncertainties, bin_edges, right=True) - 1
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)  # Ensure indices are within valid range
    # Initialize variables for UCE computation
    total_samples = uncertainties.size(0)
    uce = 0.0
    # Compute error and uncertainty per bin
    for b in range(num_bins):
        bin_mask = bin_indices == b
        bin_count = bin_mask.sum().item()
        if bin_count > 0:
            # Mean error in the bin
            bin_error = error[bin_mask].mean().item()
            # Mean uncertainty in the bin
            bin_uncertainty = uncertainties[bin_mask].mean().item()
            uce_b = abs(bin_error - bin_uncertainty)
            # print(f"Bin {b}: NumBin = {bin_count} , NumBin/sample = {bin_count/total_samples} ,Error = {bin_error}, Uncertainty = {bin_uncertainty}, uce_b = {uce_b}, uce*weight = {(bin_count / (total_samples)) * uce_b}") 
            # Update UCE
            uce += (bin_count / (total_samples)) * uce_b
    # print(f"UCE: {uce}")
    # breakpoint()
    return uce



def plot_confidence_histogram(confidence_scores, correct_predictions, epoch, args):

    # 1. Confidence Histogram
    plt.figure(figsize=(10, 5))
    plt.yscale("log")
    plt.hist(confidence_scores, bins=40, range=(0.5 , 1), alpha=0.7, color='blue', edgecolor='black')
    plt.title("Confidence Histogram")
    plt.xlabel("Predicted Confidence")
    plt.ylabel("Frequency")
    os.path.join(args.path_helper['sample_path'], 'confidence_histogram+epoch+' +str(epoch) + '.jpg')
    plt.savefig(os.path.join(args.path_helper['sample_path'], 'confidence_histogram+epoch+' +str(epoch) + '.jpg'))
    plt.close()





def plot_logit_histogram(pred_logit, epoch, args):
    # 2. Logit Histogram 
    plt.figure(figsize=(10, 5))
    # plt.yscale("log")
    plt.hist(pred_logit, bins=40, alpha=0.7, color='blue', edgecolor='black')
    plt.title("Logit Histogram")
    plt.xlabel("Predicted Logit")
    plt.ylabel("Frequency")
    os.path.join(args.path_helper['sample_path'], 'logit_histogram+epoch+' +str(epoch) + '.jpg')
    plt.savefig(os.path.join(args.path_helper['sample_path'], 'logit_histogram+epoch+' +str(epoch) + '.jpg'))
    plt.close() 


def plot_var_histogram(pred_var_ls, epoch, args):            
    # 3. Var Histogram 
    plt.figure(figsize=(10, 5))
    plt.yscale("log")
    plt.hist(pred_var_ls, bins=40, alpha=0.7, color='blue', edgecolor='black')
    plt.title("Var Histogram")
    plt.xlabel("Predicted Var")
    plt.ylabel("Frequency")
    os.path.join(args.path_helper['sample_path'], 'Var_histogram+epoch+' +str(epoch) + '.jpg')
    plt.savefig(os.path.join(args.path_helper['sample_path'], 'Var_histogram+epoch+' +str(epoch) + '.jpg'))
    plt.close() 





def plot_alpha_histogram(pred_ls_a, epoch, args):                
    # 4. alpha Histogram 
    plt.figure(figsize=(10, 5))
    plt.yscale("log")
    plt.hist(pred_ls_a, bins=40, alpha=0.7, color='blue', edgecolor='black')
    plt.title("alpha Histogram")
    plt.xlabel("Predicted alpha")
    plt.ylabel("Frequency")
    os.path.join(args.path_helper['sample_path'], 'alpha_histogram+epoch+' +str(epoch) + '.jpg')
    plt.savefig(os.path.join(args.path_helper['sample_path'], 'alpha_histogram+epoch+' +str(epoch) + '.jpg'))
    plt.close() 





def plot_beta_histogram(pred_ls_b, epoch, args):
    # 5. beta Histogram 
    plt.figure(figsize=(10, 5))
    plt.yscale("log")
    plt.hist(pred_ls_b, bins=40, alpha=0.7, color='blue', edgecolor='black')
    plt.title("beta Histogram")
    plt.xlabel("Predicted beta")
    plt.ylabel("Frequency")
    os.path.join(args.path_helper['sample_path'], 'beta+epoch+' +str(epoch) + '.jpg')
    plt.savefig(os.path.join(args.path_helper['sample_path'], 'beta_histogram+epoch+' +str(epoch) + '.jpg'))
    plt.close() 
    
    
def calculate_pearson(loss, pred_var_ls):
    # breakpoint()
    # cov = (loss - loss.mean(axis=1, keepdims=True)) * (pred_var_ls - pred_var_ls.mean(axis=1, keepdims=True))
    # pearson_corr = cov.mean(axis=1) / (loss.std(axis=1, unbiased=False) * pred_var_ls.std(axis=1, unbiased=False) + 1e-8)
    # pearson_corr_mean = pearson_corr.mean()

    
    # print(f"Average Pearson correlation per image: {pearson_corr_mean}")
    loss.flatten(start_dim=0)
    pred_var_ls.flatten(start_dim=0)        
    cov = (loss - loss.mean(axis=0, keepdims=True)) * (pred_var_ls - pred_var_ls.mean(axis=0, keepdims=True))
    pearson_corr = cov.mean(axis=0) / (loss.std(axis=0, unbiased=False) * pred_var_ls.std(axis=0, unbiased=False) + 1e-8)
    pearson_corr_mean = pearson_corr.mean()
    
    # breakpoint()
    return pearson_corr_mean

def vis_image_val(args, imgs, pred, masks, pred_var, name, epoch, reverse=False, points=None):
    # compute entropy map
    x = torch.sigmoid(pred)
    x = -x*torch.log(x + 1e-8) - (1 - x) * torch.log(1 - x + 1e-8)
    x = (x - x.amin(dim=(-1, -2), keepdim=True)) / (x.amax(dim=(-1, -2), keepdim=True) - x.amin(dim=(-1, -2), keepdim=True))
    x_ = mae(torch.sigmoid(pred), masks)
    x_ = (x_ - x_.amin(dim=(-1, -2), keepdim=True)) / (x_.amax(dim=(-1, -2), keepdim=True) - x_.amin(dim=(-1, -2), keepdim=True))
    namecat = 'Test'
    for na in name[:2]:
        img_name = na.split('/')[-1].split('.')[0]
        namecat = namecat + img_name + '+'
    pred_var_normalize = (pred_var- pred_var.amin(dim=(-1, -2), keepdim=True)) / (pred_var.amax(dim=(-1, -2), keepdim=True) - pred_var.amin(dim=(-1, -2), keepdim=True))
    # breakpoint()
    vis_image(imgs, pred, masks, x, x_, pred_var_normalize = pred_var_normalize, save_path=os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False)
    # vis_image(imgs, pred_var_normalize, masks, x, x_, save_path=os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '_var.jpg'), reverse=False, points=showp)
# breakpoint()