#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:2
#SBATCH --qos=gpu
#SBATCH --time 23:59:59
#SBATCH --mem=32gb
#SBATCH --ntasks-per-node 10


# Charge resources to account   
#SBATCH --account t_2024_dlagm

echo $SLURM_JOB_NODELIST

echo  #OMP_NUM_THREADS : $OMP_NUM_THREADS

module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate mqtth3w

# Adam, dice loss
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name Adam --model_mame UNet_Adam_dice_lr_1_e5

# Adam, BCE loss
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name Adam --model_mame UNet_Adam_BCE_lr_1_e5 --loss BCE

# Adam, combo loss
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name Adam --model_mame UNet_Adam_combo_lr_1_e5 --loss combo

# RSMprop, dice loss
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name RSMprop --model_mame UNet_RSMprop_dice_lr_1_e5 --opt RSMprop

# RSMprop, BCE loss
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name RSMprop --model_mame UNet_RSMprop_BCE_lr_1_e5 --opt RSMprop --loss BCE

# RSMprop, combo loss
python ./main.py --dataset_path /hpc/archive/T_2024_DLAGM/matteo.gianvenuti/ \
--checkpoint_path /hpc/group/T_2024_DLAGM/matteo.gianvenuti/checkpoints \
--run_name RSMprop --model_mame UNet_RSMprop_combo_lr_1_e5 --opt RSMprop --loss combo

conda deactivate