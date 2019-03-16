#!/bin/bash
#SBATCH --account=def-functor
#SBATCH --gres=gpu:1
#SBATCH --mem=4000
#SBATCH --time=0-01:30
source startup.sh
python model.py -e ${NUM_EPOCHS:-50} -d '../datasets'  ${RUN_FLAGS:-''}

