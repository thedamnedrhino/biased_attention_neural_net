#!/bin/bash
#SBATCH --account=def-functor
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --time=0-00:50
#SBATCH --output=%x_%a.out
#SBATCH --mail-user=fsharifb@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-159

# %x is the job name

NETS=(featNRO_R featNRO_S featNRO_Th featNPO_R featNPO_S featNPO_Th featNRO_Softmax_g featNRO_Softmax_ch featNPO_Softmax_g featNPO_Softmax_ch)
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
let NET_INDEX=${JOB_NUM}/4

NET=${NETS[${NET_INDEX}]}
NON_LINEAR=${NON_LINEARS[${NON_LINEAR_INDEX}]}
FREEZE=${FREEZES[${FREEZE_INDEX}]}
FREEZE_TEXT=${FREEZE_TEXTS[${FREEZE_INDEX}]}
AUGMENT=${AUGMENTS[${AUGMENT_INDEX}]}
AUGMENT_TEXT=${AUGMENT_TEXTS[${AUGMENT_INDEX}]}

FILE_NAME=${NET}_${NON_LINEAR}_${FREEZE_TEXT}_${AUGMENT_TEXT}

source startup.sh
python model.py -e 120 -d '../datasets' ${AUGMENT} -m ${FILE_NAME}.model  -l 'outputs/regular.model' -x -n ${NET} --net-args nonlinear=${NON_LINEAR} ${FREEZE} > ${FILE_NAME}.out

