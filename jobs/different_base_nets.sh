#!/bin/bash
#SBATCH --account=def-functor
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH --time=0-01:00
#SBATCH --output=%x_%a.out
#SBATCH --error=%x_%a.err
#SBATCH --mail-user=fsharifb@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-17

# %x is the job name

NETS=(two_fc diff_fc)
AGGREGATE_FEATURE_COUNTS=(18 36 72)
NON_LINEARS=(relu sigmoid tanh)
JOB_NUM=${SLURM_ARRAY_TASK_ID}
let NON_LINEAR_INDEX=${JOB_NUM}%3
let JOB_NUM=${JOB_NUM}/3
let AGGREGATE_FEATURE_COUNT_INDEX=${JOB_NUM}%3
let NET_INDEX=${JOB_NUM}/3

NET=${NETS[${NET_INDEX}]}
AGGREGATE_FEATURE_COUNT=${AGGREGATE_FEATURE_COUNTS[${AGGREGATE_FEATURE_COUNT_INDEX}]}
NON_LINEAR=${NON_LINEARS[${NON_LINEAR_INDEX}]}

OUTPUT_FOLDER=${OUTPUT_FOLDER:-'experiment_outputs/different_base_nets'}
mkdir -p ${OUTPUT_FOLDER}
FILE_NAME=${OUTPUT_FOLDER}/${NET}_${NON_LINEAR}_agg_feat-${AGGREGATE_FEATURE_COUNT}

source startup.sh
python model.py -e 120 -d '../datasets' -a -m ${FILE_NAME}.model  --base-net=${NET} -f ${AGGREGATE_FEATURE_COUNT} --non-linear=${NON_LINEAR} > ${FILE_NAME}.out

