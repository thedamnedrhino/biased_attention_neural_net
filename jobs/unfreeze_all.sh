#!/bin/bash
#SBATCH --account=def-functor
#SBATCH --gres=gpu:1
#SBATCH --mem=4000
#SBATCH --time=0-01:30
#SBATCH --output=unfreeze_all.out
#SBATCH --mail-user=fsharifb@sfu.ca
#SBATCH --mail-type=ALL

# %x is the job name

NETS=(featNRO_R featNRO_S featNRO_Th featNPO_R featNPO_S featNPO_Th)
NON_LINEARS=(none relu sigmoid tanh)
# JOB_NUM=${SLURM_ARRAY_TASK_ID}
# set job num to choose featNRO_Th and relu = 2*6 + 1
JOB_NUM=13
let NON_LINEAR_INDEX=${JOB_NUM}%6
let NET_INDEX=${JOB_NUM}/6

NET=${NETS[${NET_INDEX}]}
NON_LINEAR=${NON_LINEARS[${NON_LINEAR_INDEX}]}

source startup.sh
python model.py -e 120 -d '../datasets' -a -m ${NET}_${NON_LINEAR}.model  -l 'outputs/regular.model' -x -n ${NET} --net-args nonlinear=${NON_LINEAR} --unfreeze-all

