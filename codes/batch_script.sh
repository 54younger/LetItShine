#!/bin/bash
#SBATCH -A berzelius-2024-xxx
#SBATCH --gpus 1
#SBATCH -t 2-00:00:00
#SBATCH -C fat

python main.py --mode MM --channel 7 --fusion_mode I --mixup --name cafnet --batch_size 128 --retrain --pretrained --run_name shift_4 --fold 1
