#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100_40g:2
#SBATCH --qos=gpu
#SBATCH --time 23:59:59
#SBATCH --mem=64gb
#SBATCH --ntasks-per-node 10

#SBATCH --mail-type=ALL
#SBATCH --mail-user=matteo.gianvenuti@studenti.unipr.it

# Charge resources to account   
#SBATCH --account t_2024_dlagm

echo $SLURM_JOB_NODELIST

echo  #OMP_NUM_THREADS : $OMP_NUM_THREADS

module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate mqtth3w

<< 'COMMENT'

# Adam, dice loss (always used batch size 16, weight deacy 1e-8)
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name UNet_Adam_dice --model_name UNet_Adam_dice
# Adam, CE loss
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name UNet_Adam_CE --model_name UNet_Adam_CE --loss CE
# Adam, combo loss
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name UNet_Adam_combo --model_name UNet_Adam_combo --loss combo
# RSMprop, dice loss
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name UNet_RSMprop_dice --model_name UNet_RSMprop_dice --opt RSMprop
# RSMprop, CE loss
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name UNet_RSMprop_CE --model_name UNet_RSMprop_CE --opt RSMprop --loss CE
# RSMprop, combo loss
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name UNet_RSMprop_combo --model_name UNet_RSMprop_combo --opt RSMprop --loss combo

# With the prev setups I demostrated that the best optimized and the best loss for this task are Adam and Dice loss.
# As expected, because Adam is more robust and dynamical. So, I will use these also for the other models.
# Adam is a combination of RMSProp and Momentum so it is obviously better than RMSProp, anyway I wanted to try it 
# because a strong implementation of PyTorch-UNet (milesial/Pytorch-UNet cited in Prati's slides) use it, 
# that implementation scored 0.988423 in Dice coefficent.
# I did not consider the SGD because of its static nature. RMSProp is an improvement of SGD.
# The Dice loss measures the overlap between the img and the mask, so this will lead to a better IoU.
# Even if the Cross-entropy loss measures the correct prediction of the three classes backround, pet edge and pet
# is less appropriate for a segmentation mask task, I expected the Dice outperform it because it is widely similar to the IoU.
# I gave the proof of the Adam and Dice loss choice, the best metrics for this run are saved in runs/best_res_1run.txt

# for the next tests I improved the program by adding the dropout and another data augmentation (color jitter)


# dropout 0.15, batch size 32, learning rate 3e-4 (always weight decay 1e-6)
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name UNet_dr015_bs32_lr3e_4 --model_name UNet_dr015_bs32_lr3e_4 \
--batch_size 32 --lr 3e-4 --dropout 0.15
# dropout 0.1, batch size 32, learning rate 3e-4
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name UNet_dr01_bs32_lr3e_4 --model_name UNet_dr01_bs32_lr3e_4 \
--lr 3e-4 --batch_size 32  
# dropout 0.15, batch size 16, learning rate 2e-4
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name UNet_dr015_bs16_lr2e_4 --model_name UNet_dr015_bs16_lr2e_4 \
--dropout 0.15
# dropout 0.1, batch size 16, learning rate 2e-4
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name UNet_dr01_bs16_lr2e_4 --model_name UNet_dr01_bs16_lr2e_4
# dropout 0.15, batch size 8, learning rate 1e-4
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name UNet_dr015_bs8_lr1e_4 --model_name UNet_dr015_bs8_lr1e_4 \
--batch_size 8 --lr 1e-4 --dropout 0.15
# dropout 0.1, batch size 8, learning rate 1e-4
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name UNet_dr01_bs8_lr1e_4 --model_name UNet_dr01_bs8_lr1e_4 \
--batch_size 8 --lr 1e-4

# The best configuration is UNet_dr01_bs32_lr3e_4 with IoU 0.8377 but the configuration 
# UNet_dr015_bs16_lr2e_4 reached 0.8376 with the same L1 distance 0.0615.
# With tensorboard is possible to see that the first had better results from the beginning 
# and faster times, so the best batch size is 32, with dropout 0.1. 
# The best metrics for this run are saved in runs/best_res_2run.txt

COMMENT

# learning rate tests ("grid" search)
# lr 1e-3
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name UNet_lr1e_3 --model_name UNet_lr1e_3 --lr 1e-3
# lr 2e-3
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name UNet_lr2e_3 --model_name UNet_lr2e_3 --lr 2e-3
# lr 5e-4
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name UNet_lr5e_4 --model_name UNet_lr5e_4 --lr 5e-4
# lr 7e-5
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name UNet_lr7e_5 --model_name UNet_lr7e_5 --lr 7e-5

conda deactivate