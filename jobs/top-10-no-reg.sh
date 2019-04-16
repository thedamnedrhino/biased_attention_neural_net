#!/bin/bash
#SBATCH --account=def-functor
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --time=0-01:0
#SBATCH --output=%x_%a.out
#SBATCH --mail-user=fsharifb@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-5183

# %x is the job name

OUTPUT_FOLDER=full_diff_base_nets_outputs

BASE_NETS=(two_fc diff_fc)
NETS=(reg fcN featNRO_R featNRO_S featNRO_Th featNPO_R featNPO_S featNPO_Th featNRO_Smax_g featNRO_Smax_ch featNPO_Smax_g featNPO_Smax_ch)
REGULARIZATION_TYPES=(l1 l2 cos)
REGULARIZATION_RATES=(0.001 0.01 0.1 1 10 100)
NON_LINEARS=(relu sigmoid tanh)
FREEZES=('' '--unfreeze-all')
AUGMENTS=('' '-a')

FREEZE_TEXTS=(freeze unfreeze)
AUGMENT_TEXTS=(no_augment augment)

JOB_NUM=${SLURM_ARRAY_TASK_ID}
let AUGMENT_INDEX=${JOB_NUM}
let FREEZE_INDEX=${JOB_NUM}
let NON_LINEAR_INDEX=${JOB_NUM}
let REGULARIZATION_RATE_INDEX=${JOB_NUM}
let REGULARIZATION_TYPE_INDEX=${JOB_NUM}
let NET_INDEX=${JOB_NUM}
let BASE_NET_INDEX=${JOB_NUM}

BASE_NET=${BASE_NETS[${BASE_NET_INDEX}]}
NET=${NETS[${NET_INDEX}]}
REGULARIZATION_TYPE=${REGULARIZATION_TYPES[${REGULARIZATION_TYPE_INDEX}]}
REGULARIZATION_RATE=${REGULARIZATION_RATES[${REGULARIZATION_RATE_INDEX}]}
NON_LINEAR=${NON_LINEARS[${NON_LINEAR_INDEX}]}
FREEZE=${FREEZES[${FREEZE_INDEX}]}
FREEZE_TEXT=${FREEZE_TEXTS[${FREEZE_INDEX}]}
AUGMENT=${AUGMENTS[${AUGMENT_INDEX}]}
AUGMENT_TEXT=${AUGMENT_TEXTS[${AUGMENT_INDEX}]}

CHECKPOINT=experiment_outputs/different_base_nets/${BASE_NET}_${NON_LINEAR}.model
FILE_NAME=${BASE_NET}_${NET}_${NON_LINEAR}_${FREEZE_TEXT}_${AUGMENT_TEXT}_regul-${REGULARIZATION_TYPE}-at-${REGULARIZATION_RATE}

source startup.sh
mkdir -p ${OUTPUT_FOLDER}
python model.py -e 120 -d '../datasets' ${AUGMENT} -m ${OUTPUT_FOLDER}/${FILE_NAME}.model  -l ${CHECKPOINT} --non-linear=${NON_LINEAR} -x --base-net=${BASE_NET} -n ${NET} --net-args nonlinear=${NON_LINEAR} regularization_type=${REGULARIZATION_TYPE} regularization_rate=${REGULARIZATION_RATE} ${FREEZE} > ${OUTPUT_FOLDER}/${FILE_NAME}.out
