#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job­name="test"
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
module load cuda/9.0.176
#module load cudnn/v5.1
#module load pytorch/0.2.0p3-py27
#module load pytorch/0.3.0-py27
#module load pytorch/0.4.0-py27
module load pytorch/0.4.0-py36-cuda90
module load python/3.6.1
module load torchvision/0.2.1-py36
python mainpro_FER.py --model VGG19 --bs 128 --lr 0.01