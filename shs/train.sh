#!/bin/bash
echo "Seed $2"
CUDA_VISIBLE_DEVICES=$1 python train.py -net sam -mod sam_adapt -exp_name msa_test_isic -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 4 -dataset isic -data_path data/isic -seed $2