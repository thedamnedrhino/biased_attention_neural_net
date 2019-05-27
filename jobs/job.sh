#!/bin/bash
#SBATCH --account=def-functor
#SBATCH --gres=gpu:1
#SBATCH --mem=6000
#SBATCH --time=0:45:0
#SBATCH --output=temp/%x.out
#SBATCH --mail-user=fsharifb@sfu.ca
#SBATCH --mail-type=ALL
# %x is the job name
mkdir -p temp

OUTPUT_FOLDER=running/${SLURM_JOB_NAME}
mkdir -p $OUTPUT_FOLDER

source startup.sh
python model.py -e 120 -d '../datasets' -a -m $OUTPUT_FOLDER/$SLURM_JOB_NAME.model  -l ${MODEL_NAME:-'outputs/regular.model'} > ${OUTPUT_FOLDER}/$SLURM_JOB_NAME.out

