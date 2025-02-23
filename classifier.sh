#!/bin/bash

#SBATCH --array=1-50
#SBATCH --mem=50g
#SBATCH --time=100:00:00

module load Conda/3
conda activate py_env

python classifier.py $SLURM_ARRAY_TASK_ID
