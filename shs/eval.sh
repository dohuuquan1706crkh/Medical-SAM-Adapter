#!/bin/bash
echo "Seed $2"

# MC-Dropout
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa_test_isic -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/2024-12-01_03-02-25.230343/sam-55-best-43.pth -image_size 1024 -b 4 -dataset isic -data_path data/isic -seed $2 -val_mode mc_dropout -vis 50
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa_test_isic -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/2024-12-01_03-02-33.863709/sam-80-best-44.pth -image_size 1024 -b 4 -dataset isic -data_path data/isic -seed $2 -val_mode mc_dropout -vis 50
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa_test_isic -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/2024-12-01_03-02-49.137806/sam-99-best-45.pth -image_size 1024 -b 4 -dataset isic -data_path data/isic -seed $2 -val_mode mc_dropout -vis 50

# Deep Ensemble
CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa_test_isic -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth \
-weights_ensemble "./checkpoint/sam/2024-12-01_03-02-25.230343/sam-55-best-43.pth" \
-weights_ensemble "./checkpoint/sam/2024-12-01_03-02-25.230343/sam-99-last-43.pth" \
-weights_ensemble "./checkpoint/sam/2024-12-01_03-02-33.863709/sam-80-best-44.pth" \
-weights_ensemble "./checkpoint/sam/2024-12-01_03-02-33.863709/sam-99-last-44.pth" \
-weights_ensemble "./checkpoint/sam/2024-12-01_03-02-49.137806/sam-99-best-45.pth" \
-weights_ensemble "./checkpoint/sam/2024-12-01_03-02-49.137806/sam-95-last-45.pth" \
-image_size 1024 -b 4 -dataset isic -data_path data/isic -seed $2 -val_mode deep_ensemble -vis 50