LOSSES=('evidential' 'evidential_Dice')

for LOSS in "${LOSSES[@]}"; do
python train.py -net sam -mod sam_adpt -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -multimask_output 2 -loss ${LOSS} -gpu_device=0
done