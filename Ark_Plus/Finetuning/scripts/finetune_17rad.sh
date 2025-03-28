
# Finetune on top of Ark's initial weight
# Using VinDr-CXR seperated into 17 radiologists
python main_classification.py --data_set VinDrCXR_17rad
--data_dir /data/jliang12/jpang12/dataset/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/
--train_list unused_handled_in_main --val_list ../dataset/VinDrCXR/VinDrCXR_test_pe_global_one.txt --test_list ../dataset/VinDrCXR/VinDrCXR_test_pe_global_one.txt
--lr 0.01 --opt sgd --epochs 200 --warmup-epochs 0 --batch_size 64 
--model swin_large_384 --init ark_plus --key teacher --img_size 896 --input_size 768 --scale_up True
--pretrained_weights [PATH_TO_ARK_MODEL]