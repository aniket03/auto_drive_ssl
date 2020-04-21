#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:p40:1
#SBATCH --mem=14000
#SBATCH --job-name=train_simclr_on_stl
#SBATCH --mail-type=END
#SBATCH --mail-user=ab8700@nyu.edu
#SBATCH --output=slurm_%j.out

python simclr_auto_train_test.py --model-type 'res50' --batch-size 32 --count-negatives 6400  --epochs 150  --lr 0.1 --tmax-for-cos-decay 50  --only-train True --experiment-name e1_simclr_auto
