#!/bin/bash
## ISIC dataset
# # normal
# echo "Seed $2"
# CUDA_VISIBLE_DEVICES=$1 python train.py -net sam -mod sam_adapt -exp_name msa_test_isic -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 4 -dataset isic -data_path data/isic -seed $2

# # bayescap
# echo "Seed $2"
# CUDA_VISIBLE_DEVICES=$1 python train.py -net sam -encoder bayescap_decoder -mod sam_adapt -exp_name msa_test_isic -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 4 -dataset isic -data_path data/isic -seed $2

## BTCV dataset
# normal
echo "Seed $2"
CUDA_VISIBLE_DEVICES=$1 python train.py -net sam -mod sam_adapt -exp_name msa-3d-sam-btcv -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 2 -dataset decathlon -thd True -chunk 2 -data_path data/ -num_sample 1 -seed $2 
