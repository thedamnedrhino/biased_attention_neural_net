#!/bin/bash
#SBATCH --account=def-functor
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH --time=0-02:00
#SBATCH --output=temp/%x_%a.out
#SBATCH --mail-user=fsharifb@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-1535

mkdir -p temp

# %x is the job name

# NETS=(reg fcN featNRO_R featNRO_S featNRO_Th featNPO_R featNPO_S featNPO_Th featNRO_Smax_g featNRO_Smax_ch featNPO_Smax_g featNPO_Smax_ch)
NETS=(featNRO_Smax_g_ch featNPO_Smax_g_ch)
BIASES=(True False)
INIT_0_WEIGHTS=(True False)
REGULARIZATION_TYPES=(l1 l2 cos)
REGULARIZATION_RATES=(0 0.001 0.01 0.1)
# disable regularization for the time being
# REGULARIZATION_TYPES=(l1)
# REGULARIZATION_RATES=(0)
NON_LINEARS=(none relu sigmoid tanh)
FREEZES=('' '--unfreeze-all')
AUGMENTS=('' '-a')

FREEZE_TEXTS=(freeze unfreeze)
AUGMENT_TEXTS=(no_augment augment)

JOB_NUM=${SLURM_ARRAY_TASK_ID}
let AUGMENT_INDEX=${JOB_NUM}%2
let JOB_NUM=${JOB_NUM}/2
let FREEZE_INDEX=${JOB_NUM}%2
let JOB_NUM=${JOB_NUM}/2
let NON_LINEAR_INDEX=${JOB_NUM}%4
let JOB_NUM=${JOB_NUM}/4
let REGULARIZATION_RATE_INDEX=${JOB_NUM}%4
let JOB_NUM=${JOB_NUM}/4
let REGULARIZATION_TYPE_INDEX=${JOB_NUM}%3
let JOB_NUM=${JOB_NUM}/3
let INIT_0_INDEX=${JOB_NUM}%2
let JOB_NUM=${JOB_NUM}/2
let BIAS_INDEX=${JOB_NUM}%2
let JOB_NUM=${JOB_NUM}/2
let NET_INDEX=${JOB_NUM}

NET=${NETS[${NET_INDEX}]}
BIAS=${BIASES[${BIAS_INDEX}]}
INIT_0=${INIT_0_WEIGHTS[${INIT_0_INDEX}]}
REGULARIZATION_TYPE=${REGULARIZATION_TYPES[${REGULARIZATION_TYPE_INDEX}]}
REGULARIZATION_RATE=${REGULARIZATION_RATES[${REGULARIZATION_RATE_INDEX}]}
NON_LINEAR=${NON_LINEARS[${NON_LINEAR_INDEX}]}
FREEZE=${FREEZES[${FREEZE_INDEX}]}
FREEZE_TEXT=${FREEZE_TEXTS[${FREEZE_INDEX}]}
AUGMENT=${AUGMENTS[${AUGMENT_INDEX}]}
AUGMENT_TEXT=${AUGMENT_TEXTS[${AUGMENT_INDEX}]}

FILE_NAME=${NET}_${NON_LINEAR}_${FREEZE_TEXT}_${AUGMENT_TEXT}_regul-${REGULARIZATION_TYPE}-at-${REGULARIZATION_RATE}_init_0-${INIT_0}_bias-${BIAS}
OUTPUT_FOLDER=running/smax_global_channel
mkdir -p ${OUTPUT_FOLDER}

source startup.sh
python model.py -e 120 -d '../datasets' ${AUGMENT} -m ${OUTPUT_FOLDER}/${FILE_NAME}.model  -l 'outputs/regular.model' -x -n ${NET} --net-args nonlinear=${NON_LINEAR} regularization_type=${REGULARIZATION_TYPE} regularization_rate=${REGULARIZATION_RATE} bias=${BIAS} init_0_weights=${INIT_0_WEIGHTS} ${FREEZE} > ${OUTPUT_FOLDER}/${FILE_NAME}.out

