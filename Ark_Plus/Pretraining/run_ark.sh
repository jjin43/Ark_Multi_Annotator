#!/bin/bash
#SBATCH --job-name=SLURM_3row_VinDrCXR

#SBATCH -N 1
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 2-00:00:00
#SBATCH -p general
#SBATCH -q public

#SBATCH -o %x_slurm_%j.out
#SBATCH -e %xslurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jjin43@asu.edu

module load mamba/latest

source activate ark

# Train Ark+ with six public datasets - original example
# python main_ark.py --data_set VinDrCXR \
# --opt sgd --warmup-epochs 20  --lr 0.3 --batch_size 50 --model swin_base --init imagenet --val_loss_metric average \
# --pretrain_epochs 200  --test_epoch 10 \
# --pretrained_weights https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth \
# --momentum_teacher 0.9  --projector_features 1376  --img_resize 256 --input_size 224 --exp_name vindr_rad1

# Train Ark+ with VinDrCXR subsetted to the 17 radiologists
# python main_ark.py \
# --opt sgd --warmup-epochs 20 --lr 0.3 --workers 4 \
# --data_set VinDrCXR_rad1 --data_set VinDrCXR_rad2 --data_set VinDrCXR_rad3 --data_set VinDrCXR_rad4 --data_set VinDrCXR_rad5 --data_set VinDrCXR_rad6 --data_set VinDrCXR_rad7 --data_set VinDrCXR_rad8 --data_set VinDrCXR_rad9 --data_set VinDrCXR_rad10 --data_set VinDrCXR_rad11 --data_set VinDrCXR_rad12 --data_set VinDrCXR_rad13 --data_set VinDrCXR_rad14 --data_set VinDrCXR_rad15 --data_set VinDrCXR_rad16 --data_set VinDrCXR_rad17 \
# --model swin_base --init imagenet --val_loss_metric average \
# --img_resize 256 --input_size 224 \
# --pretrain_epochs 200 --test_epoch 10 --batch_size 100 --momentum_teacher 0.9 \
# --pretrained_weights https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth


# Train Ark+ with VinDrCXR subsetted to the 3 rows for each image
python main_ark.py \
--opt sgd --warmup-epochs 20 --lr 0.3 --workers 4 \
--data_set VinDrCXR_row1 --data_set VinDrCXR_row2 --data_set VinDrCXR_row3 \
--model swin_base --init imagenet --val_loss_metric average \
--img_resize 256 --input_size 224 \
--pretrain_epochs 200 --test_epoch 10 --batch_size 100 --momentum_teacher 0.9 \
--pretrained_weights https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth
