#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:p40:1
#SBATCH --mem=14000
#SBATCH --job-name=train_simclr_on_stl
#SBATCH --mail-type=END
#SBATCH --mail-user=ab8700@nyu.edu
#SBATCH --output=slurm_%j.out

python simclr_auto_train_test.py --model-type 'res34' --batch-size 16 --count-negatives 6400  --epochs 32  --lr 0.1 --tmax-for-cos-decay 50  --only-train True --warm-start True --prev-trained-aux-file e1_simclr_auto_aux_epoch_90  --prev-trained-main-file e1_simclr_auto_main_epoch_90  --mem-rep-file e1_simclr_auto_activ_epoch_90.npy  --experiment-name e1_simclr_auto