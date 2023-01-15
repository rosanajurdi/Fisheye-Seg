#!/bin/bash

# Slurm submission script, 
# MPI/OpenMP/GPU job with Intel MPI/srun
# CRIHAN v 1.00 - Jan 2017 
# support@criann.fr

# Not shared resources
#SBATCH --exclusive

# Job name
#SBATCH -J "inf.run"

# Batch output file
#SBATCH --output logs/inf.o%J

# Batch error file
#SBATCH --error errors/inf.e%J

# GPUs architecture and number
# ----------------------------
# Partition (submission class)

#SBATCH --partition gpu_v100
# GPUs per compute node


## SBATCH --gres gpu:1
# ----------------------------

# Job time (hh:mm:ss)
#SBATCH --time 02:00:00

# ----------------------------
# Compute nodes number
#SBATCH --nodes 1

# MPI tasks per compute node
#SBATCH --ntasks-per-node 4

# CPUs per MPI task
# (by default, OMP_NUM_THREADS is set to the same value)
#SBATCH --cpus-per-task 6

# MPI task maximum memory (MB)
#SBATCH --mem 120000 
# ----------------------------

#SBATCH --mail-type ALL
# User e-mail address
#SBATCH --mail-user rosana_jurdi@live.com

# Compiler / MPI / GPU environments
# ---------------------------------
module load compilers/intel/2017
module load mpi/intelmpi/2017
# module load cuda/9.0
module load python3-DL/torch/1.2.0-cuda10

# ---------------------------------
python3 /home/2017011/reljur01/WoodScape_Segmentation_Project/inference_Syn.py

