#!/bin/bash

#SBATCH -o "./hpc_output.log"   # Output-File -> somehow SLURM seems to be buffering stuff, not updating this file in "realtime"
#SBATCH -D .                    # Working Directory
#SBATCH -J hpc-tutorial     	# Job Name
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:tesla:2	# request two GPUs

#SBATCH --time=09:00:00 # expected runtime
#SBATCH --partition=gpu

#Job-Status per Mail:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mayet@campus.tu-berlin.de

#Using base environment.
# source activate $1
python $2 | tee > hpc.log


# To run do
# sbatch hpc_gpu_job.sh ./train.sh
