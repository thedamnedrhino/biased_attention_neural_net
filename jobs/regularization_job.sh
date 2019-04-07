#!/bin/bash
#SBATCH --account=def-functor
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --time=0-01:30
#SBATCH --output=%x_%a.out
#SBATCH --mail-user=fsharifb@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-107

# %x is the job name

NETS=(reg fcN)
TYPES=(l1 l2 l1/2)
RATES=(0.001 0.01 0.1 1 10 100)
NON_LINEARS=(relu sigmoid tanh)

JOB_NUM=${SLURM_ARRAY_TASK_ID}
let NON_LINEAR_INDEX=${JOB_NUM}%3
let JOB_NUM=${JOB_NUM}/3
let RATE_INDEX=${JOB_NUM}%6
let JOB_NUM=${JOB_NUM}/6
let TYPE_INDEX=${JOB_NUM}%3
let NET_INDEX=${JOB_NUM}/3

NET=${NETS[${NET_INDEX}]}
TYPE=${TYPES[${TYPE_INDEX}]}
RATE=${RATE[${RATE_INDEX}]}
NON_LINEAR=${NON_LINEARS[${NON_LINEAR_INDEX}]}

source startup.sh
python model.py -e 120 -d '../datasets' -a -m regularized_${NET}_${NON_LINEAR}_at-${TYPE}-${RATE}.model  -l 'outputs/regular.model' -x -n ${NET} --net-args nonlinear=${NON_LINEAR} regularization_type=${TYPE} regularization_rate=${RATE} > regularized_${NET}_${NON_LINEAR}_at-${TYPE}-${RATE}.out

