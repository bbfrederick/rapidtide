#!/bin/bash
#SBATCH -c 1
#SBATCH -t 16:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -o deeplearn_%j.out                 # File to which STDOUT will be written, including job ID
#SBATCH -e deeplearn_%j.err                 # File to which STDERR will be written, including job ID
#SBATCH --mail-type=ALL                    # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=blaise.frederick@gmail.com   # Email to which notifications will be sent

 
#source tensorflow/bin/activate

#module load gcc/6.2.0
module load cuda91
#module load python/3.6.0

python main.py
