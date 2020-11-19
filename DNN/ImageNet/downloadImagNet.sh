#!/bin/bash

#SBATCH --nodes=1

#SBATCH --gres=gpu:3cp
#SBATCH --partition=gpu
#SBATCH --mem=128G
#SBATCH --time=02:30:02
#SBATCH --output=job.%J.out
#SBATCH --error=job.%J.err
#SBATCH --job-name="example job"
#srun --nodes 1 --tasks 1 --cpus-per-task=8 --mem=128G --time 20:00:00 --pty bash

module load wget

wget http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_val.tar