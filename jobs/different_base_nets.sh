#!/bin/bash
#SBATCH --account=def-functor
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --time=0-01:30
#SBATCH --output=%x_%a.out
#SBATCH --mail-user=fsharifb@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-5

# %x is the job name

NETS=(two_fc diff_fc)
NON_LINEARS=(relu sigmoid tanh)
JOB_NUM=${SLURM_ARRAY_TASK_ID}
let NON_LINEAR_INDEX=${JOB_NUM}%3
let NET_INDEX=${JOB_NUM}/3

NET=${NETS[${NET_INDEX}]}
NON_LINEAR=${NON_LINEARS[${NON_LINEAR_INDEX}]}

source startup.sh
python model.py -e 120 -d '../datasets' -a -m ${NET}_${NON_LINEAR}.model  --base-net=${NET} --non-linear=${NON_LINEAR} > ${NET}_${NON_LINEAR}.out

