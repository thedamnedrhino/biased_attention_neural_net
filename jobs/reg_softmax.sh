#!/bin/bash
#SBATCH --account=def-functor
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --time=0-01:30
#SBATCH --output=%x_slurm.out
#SBATCH --mail-user=fsharifb@sfu.ca
#SBATCH --mail-type=ALL

# %x is the job name

source startup.sh
python model.py -e 120 -d '../datasets' -a -m reg_softmax.model  -l 'outputs/regular.model' -x -n reg --non-linear=softmax -u > reg_softmax.out

