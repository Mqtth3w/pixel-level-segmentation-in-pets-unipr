#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100_40g:3
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

# Here I directly use the best configuration finded previously for soem general things
# like the optimizer (Adam), the loss (Dice loss), etc. The dropout here is already defined 
# from who created the Net (intrinsic). So I search for the best lr. This is a SOTA model,
# and it works better with higher resolution imgs then since it recovers execution time 
# by not testing again loss and similar I use it to work on higher resolution images

COMMENT

# learning rate tests ("grid" search)
# lr 1e-3
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name DeepLabV3ResNet_lr1e_3 --model_name DeepLabV3ResNet_lr1e_3 --lr 1e-3
# lr 2e-3
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name DeepLabV3ResNet_lr2e_3 --model_name DeepLabV3ResNet_lr2e_3 --lr 2e-3
# lr 2e-4
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name DeepLabV3ResNet_lr2e_4 --model_name DeepLabV3ResNet_lr2e_4
# lr 5e-4
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name DeepLabV3ResNet_lr5e_4 --model_name DeepLabV3ResNet_lr5e_4 --lr 5e-4
# lr 7e-5
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name DeepLabV3ResNet_lr7e_5 --model_name DeepLabV3ResNet_lr7e_5 --lr 7e-5

conda deactivate