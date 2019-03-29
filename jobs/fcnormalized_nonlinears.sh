#!/bin/bash
#SBATCH --account=def-functor
#SBATCH --gres=gpu:1
#SBATCH --mem=4000
#SBATCH --time=0-01:30
#SBATCH --output=%x_%a.out
#SBATCH --mail-user=fsharifb@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-3

# %x is the job name

NON_LINEARS=(none relu sigmoid tanh)
JOB_NUM=${SLURM_ARRAY_TASK_ID}
let NON_LINEAR_INDEX=${JOB_NUM}

NON_LINEAR=${NON_LINEARS[${NON_LINEAR_INDEX}]}

source startup.sh
python model.py -e 120 -d '../datasets' -a -m fcN_${NON_LINEAR}.model  -l 'outputs/regular.model' -x -n fcN --net-args nonlinear=${NON_LINEAR}

