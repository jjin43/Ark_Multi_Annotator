
# Finetune on top of Ark's initial weight
# Using VinDr-CXR seperated into 17 radiologists
~/.conda/envs/ark_ft/bin/python main_classification.py \
--data_set VinDrCXR_17rad --data_dir /data/jliang12/jpang12/dataset/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/ \
--train_list unused_handled_in_main --val_list ../dataset/VinDrCXR/VinDrCXR_test_pe_global_one.txt --test_list ../dataset/VinDrCXR/VinDrCXR_test_pe_global_one.txt \
--lr 0.01 --opt sgd --epochs 200 --warmup-epochs 0 --batch_size 64 \
--model swin_base --init ark_plus --key teacher --img_size 256 --input_size 224 --scale_up True \
--pretrained_weights /data/jliang12/shared/pretrained_models/Ark_models/ark6_swinbase_224_ep200.pth.tar

# Ark's initial weights root path
# /data/jliang12/shared/pretrained_models/Ark_models/

# Swin Base
# ark6_swinbase_224/ep200.pth.tar

# Swin Large
# ark6_swinlarge_768/ep50.pth.tar
