gpu=0
subfix=-c1
python3 test.py --train_dataset ../dataset$subfix/train --val_dataset ../dataset$subfix/test --checkpoint_dir ./checkpoint$subfix/DeepLab_16_240_240 --img_dir ../dataset$subfix/test/images --rst_dir ./test-rsts$subfix --gt_dir ../dataset$subfix/test/labels --rst_file test_rst$subfix.txt --gpu $gpu
python3 test_mRSE.py --train_dataset ../dataset$subfix/train --val_dataset ../dataset$subfix/test --checkpoint_dir ./checkpoint$subfix/DeepLab_16_240_240 --img_dir ../dataset$subfix/test/images --rst_dir ./test-rsts$subfix --gt_dir ../dataset$subfix/test/labels --rst_file test_multi_angle$subfix.txt --gpu $gpu

subfix=-c2
python3 test.py --train_dataset ../dataset$subfix/train --val_dataset ../dataset$subfix/test --checkpoint_dir ./checkpoint$subfix/DeepLab_16_240_240 --img_dir ../dataset$subfix/test/images --rst_dir ./test-rsts$subfix --gt_dir ../dataset$subfix/test/labels --rst_file test_rst$subfix.txt --gpu $gpu
python3 test_mRSE.py --train_dataset ../dataset$subfix/train --val_dataset ../dataset$subfix/test --checkpoint_dir ./checkpoint$subfix/DeepLab_16_240_240 --img_dir ../dataset$subfix/test/images --rst_dir ./test-rsts$subfix --gt_dir ../dataset$subfix/test/labels --rst_file test_multi_angle$subfix.txt --gpu $gpu

subfix=-c3
python3 test.py --train_dataset ../dataset$subfix/train --val_dataset ../dataset$subfix/test --checkpoint_dir ./checkpoint$subfix/DeepLab_16_240_240 --img_dir ../dataset$subfix/test/images --rst_dir ./test-rsts$subfix --gt_dir ../dataset$subfix/test/labels --rst_file test_rst$subfix.txt --gpu $gpu
python3 test_mRSE.py --train_dataset ../dataset$subfix/train --val_dataset ../dataset$subfix/test --checkpoint_dir ./checkpoint$subfix/DeepLab_16_240_240 --img_dir ../dataset$subfix/test/images --rst_dir ./test-rsts$subfix --gt_dir ../dataset$subfix/test/labels --rst_file test_multi_angle$subfix.txt --gpu $gpu
