#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --qos=gpu
#SBATCH --time 23:59:59
#SBATCH --mem=80gb
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

# Here I directly use the best configuration finded previously for soem general things
# like the optimizer (Adam), the loss (Dice loss), etc. The dropout here is already defined 
# from who created the Net (intrinsic). So I search for the best lr. This is a SOTA model,
# and it works better with higher resolution imgs then since it recovers execution time 
# by not testing again loss and similar I use it to work on higher resolution images

# learning rate tests ("grid" search)
# lr 2e-4
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name DeepLabV3ResNet_lr2e_4 --model_name DeepLabV3ResNet_lr2e_4 --lr 2e-4
# lr 3e-4
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name DeepLabV3ResNet_lr3e_4 --model_name DeepLabV3ResNet_lr3e_4
# lr 4e-4
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name DeepLabV3ResNet_lr4e_4 --model_name DeepLabV3ResNet_lr4e_4 --lr 4e-4
# lr 5e-4
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name DeepLabV3ResNet_lr5e_4 --model_name DeepLabV3ResNet_lr5e_4 --lr 5e-4
# lr 7e-5
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name DeepLabV3ResNet_lr7e_5 --model_name DeepLabV3ResNet_lr7e_5 --lr 7e-5

# The best IoU 0.8935 and L1 distance 0.0387, achieved by DeepLabV3ResNet_lr5e_4
# Check best_res_1run.txt to see the best results of each run

COMMENT

# Weight decay tests (after each run I always set the new top values as default values)
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name DeepLabV3ResNet_wd1e_5 --model_name DeepLabV3ResNet_wd1e_5 --weight_decay 1e-5

python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name DeepLabV3ResNet_wd1e_4 --model_name DeepLabV3ResNet_wd1e_4 --weight_decay 1e-4

conda deactivate