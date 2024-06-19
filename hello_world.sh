#!/bin/bash
#
#SBATCH --job-name=hello_world
#SBATCH --output=sample_output.txt
#SBATCH --account=douglste-laf-lab
#SBATCH --partition=douglste-laf-lab, compute, unowned
#
#SBATCH --ntasks=1
#SBATCH --time=3:00
#SBATCH --mem-per-cpu=100mb
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=

echo "hello world"
srun sleep 60
srun hostname
