#!/bin/bash
#SBATCH --account=def-functor
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH --time=0-01:00
#SBATCH --output=temp/%x_%a.out
#SBATCH --mail-user=fsharifb@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-359

# %x is the job name

mkdir -p temp/

NETS=(reg fcN featNRO_R featNRO_S featNRO_Th featNPO_R featNPO_S featNPO_Th featNRO_Smax_g featNRO_Smax_ch)
REGULARIZATION_TYPES=(l1 l2 cos)
REGULARIZATION_RATES=(0.01 0.1 0)
NON_LINEARS=(none relu sigmoid tanh)
FREEZES=('' '--unfreeze-all')
AUGMENTS=('' '-a')

FREEZE_TEXTS=(freeze unfreeze)
AUGMENT_TEXTS=(no_augment augment)

NETS_C=${#NETS[@]}
NON_LINEARS_C=${#NON_LINEARS[@]}
REGULARIZATION_TYPES_C=${#REGULARIZATION_TYPES[@]}
REGULARIZATION_RATES_C=${#REGULARIZATION_RATES[@]}

JOB_NUM=${SLURM_ARRAY_TASK_ID}
# let AUGMENT_INDEX=${JOB_NUM}%2
# let JOB_NUM=${JOB_NUM}/2
# let FREEZE_INDEX=${JOB_NUM}%2
# let JOB_NUM=${JOB_NUM}/2
# let NON_LINEAR_INDEX=${JOB_NUM}%4
# let JOB_NUM=${JOB_NUM}/4
# let REGULARIZATION_RATE_INDEX=${JOB_NUM}%3
# let JOB_NUM=${JOB_NUM}/3
# let REGULARIZATION_TYPE_INDEX=${JOB_NUM}%3
# let NET_INDEX=${JOB_NUM}/3

let NON_LINEAR_INDEX=${JOB_NUM}%${NON_LINEARS_C}
let JOB_NUM=${JOB_NUM}/${NON_LINEARS_C}
let REGULARIZATION_RATE_INDEX=${JOB_NUM}%${REGULARIZATION_RATES_C}
let JOB_NUM=${JOB_NUM}/${REGULARIZATION_RATES_C}
let REGULARIZATION_TYPE_INDEX=${JOB_NUM}%${REGULARIZATION_TYPES_C}
let JOB_NUM=${JOB_NUM}/${REGULARIZATION_TYPES_C}
let NET_INDEX=${JOB_NUM}


let AUGMENT_INDEX=0
let FREEZE_INDEX=1

NET=${NETS[${NET_INDEX}]}
REGULARIZATION_TYPE=${REGULARIZATION_TYPES[${REGULARIZATION_TYPE_INDEX}]}
REGULARIZATION_RATE=${REGULARIZATION_RATES[${REGULARIZATION_RATE_INDEX}]}
NON_LINEAR=${NON_LINEARS[${NON_LINEAR_INDEX}]}
FREEZE=${FREEZES[${FREEZE_INDEX}]}
FREEZE_TEXT=${FREEZE_TEXTS[${FREEZE_INDEX}]}
AUGMENT=${AUGMENTS[${AUGMENT_INDEX}]}
AUGMENT_TEXT=${AUGMENT_TEXTS[${AUGMENT_INDEX}]}

OUTPUT_FOLDER=${OUTPUT_FOLDER:-running/${SLURM_JOB_NAME}}

mkdir -p ${OUTPUT_FOLDER}
mkdir -p ${OUTPUT_FOLDER}/outs

# FILE_NAME=${NET}_${NON_LINEAR}_${FREEZE_TEXT}_${AUGMENT_TEXT}_regul-${REGULARIZATION_TYPE}-at-${REGULARIZATION_RATE}
FILE_NAME=${SLURM_ARRAY_TASK_ID}_${NET}_${NON_LINEAR}_regul-${REGULARIZATION_TYPE}-at-${REGULARIZATION_RATE}

source startup.sh
python model.py -e 120 -d '../datasets' -m ${OUTPUT_FOLDER}/${FILE_NAME}.model  ${AUGMENT} -l 'outputs/regular.model' -x -n ${NET} --net-args nonlinear=${NON_LINEAR} regularization_type=${REGULARIZATION_TYPE} regularization_rate=${REGULARIZATION_RATE} ${FREEZE} > ${OUTPUT_FOLDER}/outs/${FILE_NAME}.out
