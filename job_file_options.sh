#!/bin/bash
#SBATCH --account=def-functor
#SBATCH --gres=gpu:1
#SBATCH --mem=4000
#SBATCH --time=0-01:30
#SBATCH --output=%x.out
#SBATCH --mail-user=fsharifb@sfu.ca
#SBATCH --mail-type=ALL
# %x is the job name

source startup.sh
python model.py  -m ${SLURM_JOB_NAME}.model `cat ${SLURM_JOB_NAME}.options`

