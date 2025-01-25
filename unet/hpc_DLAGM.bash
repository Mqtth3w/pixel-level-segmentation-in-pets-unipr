#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time 23:59:59
#SBATCH --mem=32gb
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
 

# Adam, dice loss
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


conda deactivate