#!/bin/bash
#
#SBATCH --job-name=douglste_hello_world
#SBATCH --output=sample_output.txt
#
#SBATCH --ntasks=1
#SBATCH --time=3:00
#SBATCH --mem-per-cpu=100mb
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=douglste@lafayette.edu

echo "hello world"
srun sleep 60
echo hostname
