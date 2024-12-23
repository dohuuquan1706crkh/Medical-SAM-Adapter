python train.py -net sam -mod sam_adpt -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -multimask_output 2 -gpu_device=0 -loss evidential


python train.py -net sam -mod sam_adpt -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -multimask_output 2 -gpu_device=0 -loss evidential_Dice -exp_name train-isic-evi_Dice


python val.py -net sam -mod sam_adpt -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -multimask_output 2 -gpu_device=0 -weights ./logs/msa_train_isic_2024_11_28_evi/Model/checkpoint_best.pth -exp_name val-isic-evi


python val.py -net sam -mod sam_adpt -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -multimask_output 1 -gpu_device=1 -weights ./logs/msa_train_isic_2024_12_06_Dice/Model/checkpoint_best.pth -exp_name val-isic-Dice


python val.py -net sam -mod sam_adpt -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -multimask_output 1 -gpu_device=2 -weights ./logs/msa_train_isic_2024_12_09_BCE/Model/checkpoint_best.pth -exp_name val-isic-BCE


python train.py -net sam -mod sam_adpt -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 128 -multimask_output 1 -gpu_device=0 -loss DiceCELoss -exp_name train_lidc -data_path ./data/LIDC/ -dataset LIDC -out_size 128


python train.py -net sam -mod sam_adpt -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 128 -multimask_output 2 -gpu_device=0 -loss DiceCELoss -exp_name train_refuge -data_path ./data/REFUGE-MultiRater/ -dataset REFUGE -out_size 128


python val.py -net sam -mod sam_adpt -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 128 -multimask_output 2 -gpu_device=0 -loss DiceCELoss -exp_name train_refuge -data_path ./data/REFUGE-MultiRater/ -dataset REFUGE -out_size 128 -weights ./logs/train_refuge_2024_12_10_12_07_57/Model/best_dice_checkpoint.pth


python train.py -net sam -mod sam_adpt -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -multimask_output 1 -gpu_device=1  -exp_name train-isic-Dice -loss DiceCELoss
