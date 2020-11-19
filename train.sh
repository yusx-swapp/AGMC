#!/bin/bash

srun --nodes 1 --tasks 1 --gres=gpu:3 --partition=gpu --time 21:00:00 --pty bash

#SBATCH --nodes=1 # request one node

#SBATCH --gres=gpu:3
#SBATCH --partition=gpu
#SBATCH --mem=128G # Maximum amount of memory this job will be given, try to estimate this to the best of your ability. This asks for 128 GB of ram.
#SBATCH --time=2-02:30:02 # ask that the job be allowed to run for 2 days, 2 hours, 30 minutes, and 2 seconds.
# everything below this line is optional, but are nice to have quality of life things
#SBATCH --output=job.%J.out # tell it to store the output console text to a file called job.<assigned job number>.out
#SBATCH --error=job.%J.err # tell it to store the error messages from the program (if it doesn't write them to normal console output) to a file called job.<assigned job muber>.err
#SBATCH --job-name="example job" # a nice readable name to give your job so you know what it is when you see it in the queue, instead of just numbers

cd /work/LAS/jannesar-lab/yusx/code

source activate Model_Compression

# let's make sure we're where we expect to be in the filesystem tree cd /work/LAS/whatever-lab/user/thing-im-working-on
# the commands we're running are below
python trainer.py