#!/bin/sh
#SBATCH --job-name=qutrit_qoc_x01_90_e6
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

python3 x01_180_qutrit.py