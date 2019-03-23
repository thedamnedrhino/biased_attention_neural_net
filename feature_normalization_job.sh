#!/bin/bash
#SBATCH --account=def-functor
#SBATCH --gres=gpu:1
#SBATCH --mem=4000
#SBATCH --time=0-01:30
#SBATCH --output=%x_%a.out
#SBATCH --mail-user=fsharifb@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-5

# %x is the job name

NETS=(featNRO_R featNRO_S featNRO_Th featNPO_R featNPO_S featNPO_Th)
JOB_NUM=${SLURM_ARRAY_TASK_ID}
NET=${NETS[${JOB_NUM}]}
source startup.sh
python model.py -e 120 -d '../datasets' -a -m ${NET}.model  -l 'outputs/regular.model' -x -n ${NET}

