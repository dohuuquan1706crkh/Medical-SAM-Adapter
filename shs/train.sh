#!/bin/bash
# # normal
# echo "Seed $2"
# CUDA_VISIBLE_DEVICES=$1 python train.py -net sam -mod sam_adapt -exp_name msa_test_isic -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 4 -dataset isic -data_path data/isic -seed $2

# flare dataset
# CUDA_VISIBLE_DEVICES=$1 python train.py -net sam -encoder default -mod sam_adapt -exp_name msa_test_flare22 -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 512 -out_size 512 -b 8 -dataset flare -data_path data/FLARE22 -seed $2

# lits dataset
# CUDA_VISIBLE_DEVICES=$1 python train.py -net sam -encoder default -mod sam_adapt -exp_name msa_test_lits17 -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 512 -out_size 512 -b 24 -dataset lits -data_path data/LiTS17 -seed $2

# drive dataset
# CUDA_VISIBLE_DEVICES=$1 python train.py -net sam -encoder default -mod sam_adapt -exp_name msa_test_drive -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 512 -out_size 512 -b 24 -dataset drive -data_path data/DRIVE -seed $2

# # chasedb dataset
# CUDA_VISIBLE_DEVICES=$1 python train.py -net sam -encoder default -mod sam_adapt -exp_name msa_test_chase -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 512 -out_size 512 -b 2 -dataset chasedb -data_path data/CHASEDB1 -seed $2

# fgadr dataset
CUDA_VISIBLE_DEVICES=$1 python train.py -net sam -encoder default -mod sam_adapt -exp_name msa_test_fgadr -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 512 -out_size 512 -b 2 -dataset fgadr1 -data_path data/FGADR -seed $2

# bayescap
# echo "Seed $2"
# CUDA_VISIBLE_DEVICES=$1 python train.py -net sam -encoder bayescap_decoder -mod sam_adapt -exp_name msa_test_isic -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 2 -dataset isic -data_path data/isic -seed $2