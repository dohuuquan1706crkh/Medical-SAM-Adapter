
 
 ## Requirement

 Install the environment:

 ``conda env create -f environment.yml``

 ``conda activate sam_adapt``

 Then download [SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth), and put it at ./checkpoint/sam/

 You can run:

 ``wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth``

 ``mv sam_vit_b_01ec64.pth ./checkpoint/sam``
 creat the folder if it does not exist

 ## Example Cases



 Begin Adapting! run: ``python train.py -net sam -encoder default -mod sam_adapt -exp_name *<exp_name>* -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 512 -out_size 512 -b 2 -dataset lits -data_path ./data/LiTS17/ -val_mode entropy -plot_histogram 0 -vis 400 -gpu_device 1``
 change "data_path" and "exp_name" for your own useage. you can change "exp_name" to anything you want.

 You can descrease the ``image size`` or batch size ``b`` if out of memory.

 Check cfg.py for more custom options

 ### Note: There is a mistake, encoder is actually for decoder's choice 


 Evaluation: The code can automatically evaluate the model on the test set during traing, set "--val_freq" to control how many epoches you want to evaluate once. You can also run val.py for the independent evaluation.

 Result Visualization: You can set "--vis" parameter to control how many epoches you want to see the results in the training or evaluation process.

 In default, everything will be saved at `` ./logs/`` 

You can download some of the datasets 
``chmod +x ./download_datasets.sh <dataset name>``

Adapting with uncertainty estimation! run: ``python train.py -net sam -encoder sure_decoder -mod sam_adapt -exp_name *<exp_name>* -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 512 -out_size 512 -b 2 -dataset lits -data_path ./data/LiTS17/ -val_mode entropy -plot_histogram 0 -vis 400 -gpu_device 1``

To validate distribution shift scenario while training, add ``-val_dis_shift 1`` and ``-dataset_val``, ``-data_path_val``

For example, we want to train on LiTs and test on FLARE:

``python train.py -net sam -encoder default -mod sam_adapt -exp_name msa_train_Lits_shift_FLARE -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 512 -out_size 512 -b 2 -dataset lits -data_path ./data/LiTS17/ -val_dis_shift 1 -dataset_val flare -data_path_val ./data/FLARE22/ -val_mode entropy -plot_histogram 0 -vis 400 -gpu_device 1``

### Note: Check cfg.py 

For uncertainty estimation validation, some of the baselines are included:

``-val_mode <method to estimate uncertainty>`` 
Methods: entropy, ttdac, ttdap, mc_dropout, deep_ensemble

To use deep ensemble, we need a list of independent checkpoint to test:

``python val.py -net sam -encoder default -mod sam_adapt -exp_name msa_test_FLARE2Lits -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth \
-weights_ensemble "<path to ckpt 1>" \
-weights_ensemble "<path to ckpt 2>" \
-weights_ensemble "<path to ckpt 3>" \
-weights_ensemble "<path to ckpt 4>" \
...
-weights_ensemble "<path to ckpt 10>" \
-image_size 512 -out_size 512 -b 2 -dataset lits -data_path ./data/LiTS17/ -val_mode deep_ensemble -plot_histogram False -vis 0 -gpu_device 2``
 