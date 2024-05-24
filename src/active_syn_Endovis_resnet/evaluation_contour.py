import os
import numpy as np
import cv2
import matplotlib.pyplot as plt  

def evaluate_IOU_nearContour(output_path,image_path,label_path):
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
        
        contour_mask = np.zeros_like(gt)        
        # contours,_ = cv2.findContours(gt*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # this is colab version
        _, contours, _ = cv2.findContours(gt*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # this is local version, returns 3 values
        # print(len(contours))
        cv2.drawContours(contour_mask,contours,-1,(1,1,1),20)
        
        gt = gt*contour_mask
        
        #
        T=0.5
        output = cv2.imread(os.path.join(output_path,file))
        
        try:
            output = cv2.resize(output,(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_NEAREST)
        except:
            print('[IOU_nb] file not found:')
            print(os.path.join(output_path,file))
            continue
    
        output=output[:,:,1]/255
        output = output*contour_mask    
        
        if (np.count_nonzero(output)+np.count_nonzero(gt)) is 0:
            dice = 1
            iou = 1
        else:
            dice = (2*np.count_nonzero(gt*output))/(np.count_nonzero(output)+np.count_nonzero(gt)+0.000001) 
                
            iou = np.count_nonzero(gt*output)/(np.count_nonzero(output+gt)+0.000001)
        
        
        
        dices.append(dice)
        ious.append(iou)
        names.append(file[:-4])
        
    
    mean_dice = np.mean(dices)
    mean_iou = np.mean(ious)
    
    
    print("mean_dice={},mean_iou={}".format(mean_dice,mean_iou))
    return mean_iou


    
    
    
    
    
    
    
    
    
    
    
    