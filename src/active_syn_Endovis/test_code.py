# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 08:48:28 2021

@author: 75678
"""

import numpy as np
import math
import cv2
import glob
import os
import sys
import glob
import random

import generator_utils as g_utils
import active_learner as al
from generator_utils import rand_from_range

folder_img = './data_img/raw_data_base/images'
folder_label = './data_img/raw_data_base/labels'
folder_bkgd = './data_img/background_for_syn_cadaver'

folder_rst = './data_img/result'

img_file_name_ = al.get_file_names(folder_img, 'jpg')
bkgd_file_name_ = al.get_file_names(folder_bkgd, 'jpg')

def img_inpaint(img, label, tool_extract_map, bkgd = None, bkgd_percent = 0.9, fusion_type = 'avg', fusion_blur = 15):
    # g_utils.show_img('img', img)
    # g_utils.show_img('bkgd', bkgd)
    total_pixels = label.shape[0] * label.shape[1]
    non_zero_pixels = np.count_nonzero(label)
    
    if (non_zero_pixels/total_pixels) > 0.3 or g_utils.rand_from_range(0,1)<bkgd_percent:
        tool_extract_mask = tool_extract_map
        
        tool_mask_blur = tool_extract_mask.astype(np.float32)  #+ tool_map
        tool_mask_blur = tool_mask_blur * 100

        if fusion_type == 'avg':
            tool_mask_blur = cv2.blur(tool_mask_blur, (fusion_blur, fusion_blur))
        if fusion_type == 'gauss':
            fusion_blur = g_utils.round_up_to_odd(fusion_blur)
            tool_mask_blur = cv2.GaussianBlur(tool_mask_blur, (fusion_blur,fusion_blur), fusion_blur)
        tool_extract_mask = tool_mask_blur/100
        
        tool_map_reverse = np.ones(tool_extract_mask.shape, np.float32) - tool_extract_mask
        
        img_tool_adjusted = bkgd
        img_bkgd = img
        for chn in range(0,3):
            img_tool_adjusted[:,:,chn] = img_tool_adjusted[:,:,chn] * tool_extract_mask
            img_bkgd[:,:,chn] = img_bkgd[:,:,chn] * tool_map_reverse
            
        img_new = (img_bkgd + img_tool_adjusted).clip(0,255).astype(np.uint8)
        
    else:
        img_r = cv2.rotate(img, cv2.ROTATE_180) 
        label_r = cv2.rotate(label, cv2.ROTATE_180) 
        tool_extract_map_r = cv2.rotate(tool_extract_map, cv2.ROTATE_180)
        tool_extract_mask = tool_extract_map
        
        tool_mask_blur = tool_extract_mask.astype(np.float32)  #+ tool_map
        tool_mask_blur = tool_mask_blur * 100

        if fusion_type == 'avg':
            tool_mask_blur = cv2.blur(tool_mask_blur, (fusion_blur, fusion_blur))
        if fusion_type == 'gauss':
            fusion_blur = g_utils.round_up_to_odd(fusion_blur)
            tool_mask_blur = cv2.GaussianBlur(tool_mask_blur, (fusion_blur,fusion_blur), fusion_blur)
        tool_extract_mask = tool_mask_blur/100
        
        tool_map_reverse = np.ones(tool_extract_mask.shape, np.float32) - tool_extract_mask
        
        img_tool_adjusted = img_r
        img_bkgd = img
        # g_utils.show_img('img_tool_adjusted', img_tool_adjusted)
        # g_utils.show_img('img_bkgd', img_bkgd)
        for chn in range(0,3):
            img_tool_adjusted[:,:,chn] = img_tool_adjusted[:,:,chn] * tool_extract_mask
            img_bkgd[:,:,chn] = img_bkgd[:,:,chn] * tool_map_reverse
            
        img_new = (img_bkgd + img_tool_adjusted).clip(0,255).astype(np.uint8)
        
    return img_new

for i in range(0, 50):
    img_file = random.choice(img_file_name_)
    label_file = img_file[:-3] + 'png'
    bkgd_file = random.choice(bkgd_file_name_)
    
    image = cv2.imread(folder_img + '/' + img_file)
    label = cv2.imread(folder_label + '/' + label_file, cv2.IMREAD_GRAYSCALE)
    bkgd = cv2.imread(folder_bkgd + '/' + bkgd_file)
    
    tool_img, tool_label, tool_extract_map = g_utils.extract_tool(np.copy(image), label, dilation = 40)
    
    img_new = img_inpaint(image, label, tool_extract_map, bkgd = bkgd, fusion_type = 'avg', fusion_blur = 15)
    
    cv2.imwrite(folder_rst + '/img_' + str(i) + '.jpg', img_new)
    
    

























