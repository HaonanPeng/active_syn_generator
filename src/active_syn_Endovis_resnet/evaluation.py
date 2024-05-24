import os
import numpy as np
import cv2
import matplotlib.pyplot as plt  
import argparse



def evaluate_seg_result(result_path, label_path):
    dices = []
    ious = []
    names=[]
    files=os.listdir(label_path)
    for file in files:
        if not file.endswith(".png"):
            continue
        
        #    
        gt = cv2.imread(os.path.join(label_path,file))
         
        gt = gt[:,:,0]
        gt_=1-gt
        
        #
        output = cv2.imread(os.path.join(result_path,file))
        output = cv2.resize(output,(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_NEAREST)
    
        output=output[:,:,1]/255
        output_=1-output
    
        #
        if (np.count_nonzero(output)+np.count_nonzero(gt)) is 0:
            dice = 2*(np.count_nonzero(gt_*output_))/(np.count_nonzero(output_)+np.count_nonzero(gt_)+0.000001)
            iou = np.count_nonzero(gt_*output_)/(np.count_nonzero(output_+gt_)+0.000001)
        else:
            dice = (np.count_nonzero(gt*output))/(np.count_nonzero(output)+np.count_nonzero(gt)+0.000001) \
                +(np.count_nonzero(gt_*output_))/(np.count_nonzero(output_)+np.count_nonzero(gt_)+0.000001)
            iou = 0.5*np.count_nonzero(gt*output)/(np.count_nonzero(output+gt)+0.000001) \
                +0.5*np.count_nonzero(gt_*output_)/(np.count_nonzero(output_+gt_)+0.000001)
        
        
        
        dices.append(dice)
        ious.append(iou)
        names.append(file[:-4])
    

    mean_dice = np.mean(dices)
    mean_iou = np.mean(ious)
    
    print("mean_dice={},mean_iou={}".format(mean_dice,mean_iou))

    