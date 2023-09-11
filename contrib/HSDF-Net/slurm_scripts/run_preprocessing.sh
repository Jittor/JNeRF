#!/bin/bash

#SBATCH -p cpu20
#SBATCH -o ./slurm_scripts/out/%j.out
#SBATCH -e ./slurm_scripts/out/%j.err
#SBATCH -t 1:00:00
#SBATCH -a 0-199%30
#SBATCH -c 32


python dataprocessing/preprocess.py --num_cpus 32 --num_chunks 200 --current_chunk $SLURM_ARRAY_TASK_ID
