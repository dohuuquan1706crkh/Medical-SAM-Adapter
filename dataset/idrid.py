
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from utils import generate_click_prompt, random_box, random_click


class IDRiD(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):

        self.args = args
        self.data_path = os.path.join(data_path)
        self.mode = mode
        if self.mode == 'Training':
            img_path = os.path.join(self.data_path, "train")
            msk_path = os.path.join(self.data_path, "train_labels/Haemorrhages/")
            # self.name_list = os.listdir(img_train_path)
            # self.msk_list = os.listdir(msk_train_path)
        else:
            img_path = os.path.join(self.data_path, "test")
            msk_path = os.path.join(self.data_path, "test_labels/Haemorrhages/")
        self.img_list = os.listdir(img_path)
        self.msk_list = os.listdir(msk_path)
        self.args = args

        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1
        label = 1   # the class to be segmented

        """Get the images"""
        name_img = self.img_list[index]
        name_msk = self.msk_list[index]
        if self.mode == 'Training':
            img_path = os.path.join(self.data_path, "train", name_img)
            # msk_path = os.path.join(self.data_path, "train_labels/Haemorrhages/", name_msk)
            if self.args.dataset == 'idrid1':
                msk_path1 = os.path.join(self.data_path, "train_labels/Haemorrhages/", name_msk)
                mask = np.array(Image.open(msk_path1).convert('L'))
                
            elif self.args.dataset == 'idrid2':
                msk_path2 = os.path.join(self.data_path, "train_labels/Hard Exudates/", name_msk)
                mask = np.array(Image.open(msk_path2).convert('L'))
               
            elif self.args.dataset == 'idrid3':
                msk_path3 = os.path.join(self.data_path, "train_labels/Microaneurysms/", name_msk)
                mask = np.array(Image.open(msk_path3).convert('L'))
                
            elif self.args.dataset == 'idrid4':
                msk_path4 = os.path.join(self.data_path, "train_labels/Soft Exudates/", name_msk)
                mask = np.array(Image.open(msk_path4).convert('L'))
            
        else:
            img_path = os.path.join(self.data_path, "test", name_img)
            # msk_path = os.path.join(self.data_path, "test_labels/Haemorrhages/", name_msk)
            if self.args.dataset == 'idrid1':
                msk_path1 = os.path.join(self.data_path, "test_labels/Haemorrhages/", name_msk)
                mask = np.array(Image.open(msk_path1).convert('L'))

            elif self.args.dataset == 'idrid2':
                msk_path2 = os.path.join(self.data_path, "test_labels/Hard Exudates/", name_msk)
                mask = np.array(Image.open(msk_path2).convert('L'))

            elif self.args.dataset == 'idrid3':
                msk_path3 = os.path.join(self.data_path, "test_labels/Microaneurysms/", name_msk)
                mask = np.array(Image.open(msk_path3).convert('L'))
            elif self.args.dataset == 'idrid4':
                msk_path4 = os.path.join(self.data_path, "test_labels/Soft Exudates/", name_msk)
                mask = np.array(Image.open(msk_path4).convert('L'))
            
            # msk_path = os.path.join(self.data_path, "test_labels/Hard Exudates/", name_msk)
            
        img = Image.open(img_path).convert('RGB')

        # print(img.shape)
        # mask = np.max(np.stack([mask1, mask2, mask3, mask4], axis=0), axis=0)
        mask = Image.fromarray(mask)
        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask) / 255, point_label)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask).int()
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask
        name_img = name_img.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name_img}
        # breakpoint()
        # print(type(img))
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }