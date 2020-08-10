#!/bin/bash

#name job pythondemo, output to slurm file, use partition all, run for 1500 minutes and use 40GB of ram
#SBATCH -J 'hmm_parcels'
#SBATCH -o logfiles/hmm_parcels-%j.out
#SBATCH --error=logfiles/hmm_parcels%j.err
#SBATCH -p all
#SBATCH -t 1000
#SBATCH -c 1 --mem 6000
#SBATCH --mail-type ALL
#SBATCH --mail-user jamalw@princeton.edu
#SBATCH --array=0-15

module load pyger/0.9.1
export PYTHONMALLOC=debug

python -duv /jukebox/norman/jamalw/MES/prototype/link/scripts/hmm_parcels_sl.py $SLURM_ARRAY_TASK_ID 0

python -duv /jukebox/norman/jamalw/MES/prototype/link/scripts/hmm_parcels_sl.py $SLURM_ARRAY_TASK_ID 1


