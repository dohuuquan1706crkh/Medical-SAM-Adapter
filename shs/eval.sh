#!/bin/bash
echo "Seed $2"
### ISIC dataset
# MC-Dropout
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa_test_isic -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/2024-12-01_03-02-25.230343/sam-55-best-43.pth -image_size 1024 -b 4 -dataset isic -data_path data/isic -seed $2 -val_mode mc_dropout -vis 50
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa_test_isic -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/2024-12-01_03-02-33.863709/sam-80-best-44.pth -image_size 1024 -b 4 -dataset isic -data_path data/isic -seed $2 -val_mode mc_dropout -vis 50

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

# # TTDA color jitter
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa_test_isic -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/2024-12-01_03-02-49.137806/sam-99-best-45.pth -image_size 1024 -b 4 -dataset isic -data_path data/isic -seed $2 -val_mode ttdac -vis 50

# # TTDA pixel-wise noise
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa_test_isic -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/2024-12-01_03-02-49.137806/sam-99-best-45.pth -image_size 1024 -b 4 -dataset isic -data_path data/isic -seed $2 -val_mode ttdap -vis 50

### BTCV dataset
# Normal
CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa-3d-sam-btcv -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/2025-01-02_03-29-56.420726/sam-99-last-40.pth -image_size 1024 -b 1 -dataset decathlon -data_path data/ -thd True -seed $2 -val_mode normal -num_sample 1 -chunk 2 -gpu True

# MC-Dropout
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa-3d-sam-btcv -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/2025-01-02_03-29-56.420726/sam-99-last-40.pth -image_size 1024 -b 1 -dataset decathlon -data_path data/ -thd True -seed $2 -val_mode mc_dropout -vis 50 -num_sample 1 -chunk 2 -gpu True

# # Deep Ensemble
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa_test_isic -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth \
# -weights_ensemble "./checkpoint/sam/2024-12-01_03-02-25.230343/sam-55-best-43.pth" \
# -weights_ensemble "./checkpoint/sam/2024-12-01_03-02-25.230343/sam-99-last-43.pth" \
# -weights_ensemble "./checkpoint/sam/2024-12-01_03-02-33.863709/sam-80-best-44.pth" \
# -weights_ensemble "./checkpoint/sam/2024-12-01_03-02-33.863709/sam-99-last-44.pth" \
# -weights_ensemble "./checkpoint/sam/2024-12-01_03-02-49.137806/sam-99-best-45.pth" \
# -weights_ensemble "./checkpoint/sam/2024-12-01_03-02-49.137806/sam-95-last-45.pth" \
# -image_size 1024 -b 4 -dataset isic -data_path data/isic -seed $2 -val_mode deep_ensemble -vis 50
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa-3d-sam-btcv -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth  -weights_ensemble "./checkpoint/sam/2025-01-02_03-29-56.420726/sam-99-last-40.pth" \
#  -weights_ensemble "./checkpoint/sam/2025-01-02_03-30-04.900914/sam-99-last-41.pth" \
#  -weights_ensemble "./checkpoint/sam/2025-01-02_03-30-16.510758/sam-99-last-42.pth" \
#  -image_size 1024 -b 1 -dataset decathlon -data_path data/ -thd True -seed $2 -val_mode deep_ensemble -vis 50 -num_sample 1 -chunk 2 -gpu True
# BayesCap
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -encoder bayescap_decoder -exp_name msa_test_isic -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/2024-12-20_09-09-17.852200/sam-95-best-40.pth -image_size 1024 -b 4 -dataset isic -data_path data/isic -seed $2 -val_mode bayescap -vis 50

# # TTDA color jitter
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa-3d-sam-btcv -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/2025-01-02_03-29-56.420726/sam-99-last-40.pth -image_size 1024 -b 1 -dataset decathlon -data_path data/ -thd True -seed $2 -val_mode ttdac -vis 50 -num_sample 1 -chunk 2 -gpu True

# # TTDA pixel-wise noise
# CUDA_VISIBLE_DEVICES=$1 python val.py -net sam -mod sam_adapt -exp_name msa-3d-sam-btcv -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -weights ./checkpoint/sam/2025-01-02_03-29-56.420726/sam-99-last-40.pth -image_size 1024 -b 1 -dataset decathlon -data_path data/ -thd True -seed $2 -val_mode ttdap -vis 50 -num_sample 1 -chunk 2 -gpu True