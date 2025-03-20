# Train Ark+ with six public datasets
python main_ark.py --data_set VinDrCXR \
--opt sgd --warmup-epochs 20  --lr 0.3 --batch_size 50 --model swin_base --init imagenet --val_loss_metric average \
--pretrain_epochs 200  --test_epoch 10 \
--pretrained_weights https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth \
--momentum_teacher 0.9  --projector_features 1376  --img_resize 256 --input_size 224 --exp_name vindr_rad1

