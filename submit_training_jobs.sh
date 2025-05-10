#!/bin/bash
#SBATCH --job-name="sft_dpo_train"
#SBATCH --output="sft_dpo_train.%j.%N.out"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=208G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4  # could be 1 for py-torch
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=account_name    # <- match to a "Project" returned by the "accounts" command
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 04:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

export OMP_NUM_THREADS=1  # if code is not multithreaded, otherwise set to 8 or 16
srun -N 1 -n 4 ./a.out > myjob.out
# py-torch example, --ntasks-per-node=1 --cpus-per-task=64
# srun python3 multiple_gpu.py