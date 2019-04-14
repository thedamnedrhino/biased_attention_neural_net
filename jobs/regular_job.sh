#!/bin/bash
#SBATCH --account=def-functor
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=0-01:00
#SBATCH --output=temp/regular_%a.out
#SBATCH --mail-user=fsharifb@sfu.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-19
# %x is the job name

as=('' '-a')
ats('out' '')
jn=${SLURM_JOB_NUM}
let ai=$jn%2
a=${as[$ai]}
at=${ats[$ai]}
output_dir=averages/regular/with${at}_augment/
mkdir -p temp
mkdir -p $output_dir
source startup.sh
python model.py -e 120 -d '../datasets' ${a} -m ${output_dir}/regular_${ai}.model > ${output_dir}/regular_${ai}.out

