#!/bin/bash
echo "Seed $2"

# MC-Dropout
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa_test_isic -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/2024-12-01_03-02-25.230343/sam-55-best-43.pth -image_size 1024 -b 4 -dataset isic -data_path data/isic -seed $2 -val_mode mc_dropout -vis 50
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa_test_isic -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/2024-12-01_03-02-33.863709/sam-80-best-44.pth -image_size 1024 -b 4 -dataset isic -data_path data/isic -seed $2 -val_mode mc_dropout -vis 50
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa_test_isic -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/2024-12-01_03-02-49.137806/sam-99-best-45.pth -image_size 1024 -b 4 -dataset isic -data_path data/isic -seed $2 -val_mode mc_dropout -vis 50

# # Deep Ensemble
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa_test_isic -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth \
# -weights_ensemble "./checkpoint/sam/2024-12-01_03-02-25.230343/sam-55-best-43.pth" \
# -weights_ensemble "./checkpoint/sam/2024-12-01_03-02-25.230343/sam-99-last-43.pth" \
# -weights_ensemble "./checkpoint/sam/2024-12-01_03-02-33.863709/sam-80-best-44.pth" \
# -weights_ensemble "./checkpoint/sam/2024-12-01_03-02-33.863709/sam-99-last-44.pth" \
# -weights_ensemble "./checkpoint/sam/2024-12-01_03-02-49.137806/sam-99-best-45.pth" \
# -weights_ensemble "./checkpoint/sam/2024-12-01_03-02-49.137806/sam-95-last-45.pth" \
# -image_size 1024 -b 4 -dataset isic -data_path data/isic -seed $2 -val_mode deep_ensemble -vis 50

# BayesCap
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -encoder bayescap_decoder -exp_name msa_test_isic -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/2024-12-20_09-09-17.852200/sam-95-best-40.pth -image_size 1024 -b 4 -dataset isic -data_path data/isic -seed $2 -val_mode bayescap -vis 50
# python val.py -net sam -mod sam_adapt -encoder bayescap_decoder -exp_name msa_test_isic -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./sam-bayes-cap.pth -image_size 1024 -b 2 -dataset isic -data_path data/isic -val_mode bayescap -vis 50
# # TTDA color jitter
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa_test_isic -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/2024-12-01_03-02-49.137806/sam-99-best-45.pth -image_size 1024 -b 4 -dataset isic -data_path data/isic -seed $2 -val_mode ttdac -vis 50

# # TTDA pixel-wise noise
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa_test_isic -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/2024-12-01_03-02-49.137806/sam-99-best-45.pth -image_size 1024 -b 4 -dataset isic -data_path data/isic -seed $2 -val_mode ttdap -vis 50

# # URN paper
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa_test_isic -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/2024-12-01_03-02-49.137806/sam-99-best-45.pth -image_size 1024 -b 4 -dataset isic -data_path data/isic -seed $2 -val_mode urn -vis 50

# flare
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa_test_flare -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/2025-02-28_22-28-05.223789/sam-25-last-40.pth -image_size 512 -out_size 512 -b 16 -dataset flare -data_path data/FLARE22 -seed $2 -val_mode urn -vis 50

# lits
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa_test_lits -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/2025-03-01_03-02-35.779095/sam-95-last-40.pth -image_size 512 -out_size 512 -b 16 -dataset lits -data_path data/LiTS17 -seed $2 -val_mode urn -vis 50

# # drive
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa_test_drive -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/drive/sam-85-best-40.pth -image_size 512 -out_size 512 -b 2 -dataset drive -data_path data/DRIVE -seed $2 -val_mode urn -vis 50

# # flare -> lits
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa_test_flare -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/flare/sam-25-last-40.pth -image_size 512 -out_size 512 -b 16 -dataset flare -data_path data/FLARE22 -val_dis_shift 1 -dataset_val lits -data_path_val ./data/LiTS17/ -seed $2 -val_mode urn -vis 50

# lits -> flare
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa_test_lits -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/lits/sam-95-last-40.pth -image_size 512 -out_size 512 -b 16 -dataset lits -data_path data/LiTS17 -val_dis_shift 1 -dataset_val flare -data_path_val ./data/FLARE22/ -seed $2 -val_mode urn -vis 50

# # chasedb
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa_test_drive -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/chasedb/sam-90-best-40.pth -image_size 512 -out_size 512 -b 2 -dataset chasedb -data_path data/CHASEDB1 -seed $2 -val_mode urn -vis 50

# # chasedb -> drive
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa_test_drive -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/chasedb/sam-90-best-40.pth -image_size 512 -out_size 512 -b 2 -dataset chasedb -data_path data/CHASEDB1 -val_dis_shift 1 -dataset_val drive -data_path_val ./data/DRIVE/ -seed $2 -val_mode urn -vis 50

# drive -> chasedb
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa_test_drive -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/chasedb/sam-90-best-40.pth -image_size 512 -out_size 512 -b 2 -dataset drive -data_path ./data/DRIVE/ -val_dis_shift 1 -dataset_val chasedb -data_path_val ./data/CHASEDB1 -seed $2 -val_mode urn -vis 50