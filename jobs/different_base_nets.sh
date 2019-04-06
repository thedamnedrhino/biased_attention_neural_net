#!/bin/bash
#SBATCH --account=def-functor
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --time=0-01:30
#SBATCH --output=%x_%a.out
#SBATCH --mail-user=fsharifb@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-2

# %x is the job name

BASE_NETS=(simple two_fc diff_fc)
JOB_NUM=${SLURM_ARRAY_TASK_ID}
BASE_NET=${BASE_NETS[${JOB_NUM}]}

source startup.sh
python model.py -e 120 -d '../datasets' -a -m base_${BASE_NET}.model > base_${BASE_NET}.out
