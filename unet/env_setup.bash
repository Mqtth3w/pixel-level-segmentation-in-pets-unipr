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

conda install pytorch torchvision torchaudio cudatoolkit -c pytorch

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

pip install tensorboard

python -c "import torch; print(torch.cuda.is_available())"

conda deactivate