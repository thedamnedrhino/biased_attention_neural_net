#!/bin/bash
#SBATCH --account=def-functor
#SBATCH --gres=gpu:1
#SBATCH --mem=4000
#SBATCH --time=0-01:30
#SBATCH --output=%x.out
# %x is the job name

source startup.sh
python model.py -e ${NUM_EPOCHS:-50} -d '../datasets' -a -m ${SLURM_JOB_NAME}.model  -l ${MODEL_NAME:-'outputs/regular.model'} {RUN_FLAGS:-''}

