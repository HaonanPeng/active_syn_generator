# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 21:27:07 2021

@author: 75678
"""

import numpy as np
import pandas as pd
import cv2
import time
import os
import sys
import glob
import random

import generator_utils as g_utils
import active_learner as al
from generator_utils import rand_from_range

class active_syn_generator:
    
    cur_iter = 0
    small_df = True
    
    init_bkgd = None
    generate_bkgd = None
    
    img_size = None
    
    extract_dilation = None
    
    input_img = None
    input_tool = None
    input_label = None
    input_extract_map = None
    
    random_bkgd = None
    
    tool = None # a pandas series, containing the image of the tool and relative imformation
    background = None # a pandas series, containing the image of the background and relative imformation
    
    tool_shrinked = None
    syn_img_final = None
    
    syn_img_final = None # final result of synthetic image
    syn_img_final_compressed = None
    
    folder_bkgd = []
    folder_tool = []
    folder_learn_iter = None
    
    list_tool_ = []
    list_bkgd_ = []
    
    random_tool_shape = 0
    
    output_process = False
    file_name = None
    syn_type = None
    
    
    def init_syn_img_generator(self, folder_learn_iter, extract_dilation, img_size, init_bkgd = True, generate_bkgd = False, output_process = False):
        self.folder_bkgd = []
        self.folder_tool = []
        
        self.list_tool_ = []
        self.list_bkgd_ = []
        
        self.init_bkgd = init_bkgd
        self.generate_bkgd = generate_bkgd
        self.folder_learn_iter = folder_learn_iter
        self.extract_dilation = extract_dilation
        self.img_size = img_size
        self.output_process = output_process

        self.cur_iter = 0
        self.folder_tool.append(folder_learn_iter + '/syn_tool')
        al.create_folder(self.folder_tool[self.cur_iter])
        al.create_folder(self.folder_tool[self.cur_iter] + '/images')
        al.create_folder(self.folder_tool[self.cur_iter] + '/labels')
        
        self.extract_tool_from_folder(folder_learn_iter + '/query_added', dilation = self.extract_dilation)
        self.extract_tool_from_folder(folder_learn_iter + '/labeled', dilation = self.extract_dilation)
        
        self.folder_bkgd.append(folder_learn_iter + '/background')
        
        self.tool = pd.Series({'index': None,
                                'img': None,
                                'type': None,
                                'contour': None,
                                'tool_map': None,
                                'lines': None,
                                'distance_map': None,
                                'geo_feature_points': None,
                                'geo_feature_vectors': None,
                                'render_feature_points': None,
                                'render_feature_vectors': None})
        
        self.background = pd.Series({'index': None,
                                     'img': None,
                                     'group': None})
        
        return None
    
    def init_new_iteration(self, iteration, folder_learn_iter, dilation=60):
        self.folder_learn_iter = folder_learn_iter
        
        self.cur_iter = iteration
        self.folder_tool.append(folder_learn_iter + '/syn_tool')
        al.create_folder(self.folder_tool[self.cur_iter])
        al.create_folder(self.folder_tool[self.cur_iter] + '/images')
        al.create_folder(self.folder_tool[self.cur_iter] + '/labels')
        self.extract_tool_from_folder(folder_learn_iter + '/query_added', dilation = dilation)
        self.extract_tool_from_folder(folder_learn_iter + '/labeled', dilation = dilation)
        
        self.folder_bkgd.append(folder_learn_iter + '/background')
        
        return None
    
    # generate background using image inpainting, must be executed before synthetic images are generated, otherwise it may generate a lot same backgrounds
    def generate_background(self):
        if self.generate_bkgd == False:
            return None
        
        al.copy_folder2(source_folder = self.folder_learn_iter + '/labeled', target_folder = self.folder_learn_iter + '/query_added')
        img_files_ = al.get_file_names(self.folder_learn_iter + '/query_added/images', 'jpg')
        
        for img_file in img_files_:
            label_file = img_file[:-3] + 'png'
            img = cv2.resize(cv2.imread(self.folder_learn_iter + '/query_added/images/' + img_file), self.img_size)
            label = cv2.resize(cv2.imread(self.folder_learn_iter + '/query_added/labels/' + label_file, cv2.IMREAD_GRAYSCALE), self.img_size)
            tool_extract_map = g_utils.img_dilatation(label, 0, int(self.extract_dilation/2))
            
            if self.init_bkgd == True:
                bkgd_file = random.choice(self.list_bkgd_)    
                img_bkgd_true = cv2.resize(cv2.imread(self.folder_bkgd[self.cur_iter] + '/' + bkgd_file), self.img_size)
                
                img_bkgd = g_utils.find_inpaint_rotation(img_1 = img, label_1 = label, img_2 = img, label_2 = label)           
                if img_bkgd.all() == 0 or g_utils.rand_from_range(0, 1)<0.5:
                    bkgd_file = random.choice(self.list_bkgd_)    
                    img_bkgd = cv2.resize(cv2.imread(self.folder_bkgd[self.cur_iter] + '/' + bkgd_file), self.img_size)
                else:
                    img_bkgd = img_bkgd_true           
                img_new = self.img_inpaint(img, label, tool_extract_map, bkgd = img_bkgd, fusion_type = 'avg', fusion_blur = 15, img_size = self.img_size)
                cv2.imwrite(self.folder_learn_iter + '/background/' + img_file, img_new)
            
            else:
                img_bkgd = g_utils.find_inpaint_rotation(img_1 = img, label_1 = label, img_2 = img, label_2 = label)
                if img_bkgd.any() != 0:
                    img_bkgd = img_bkgd
                
                # print(img_files_)
                img_files_shuffled_ = random.sample(img_files_, len(img_files_))
                # print(img_files_shuffled_)
                img_num = len(img_files_shuffled_)
                
                for bkgd_file in img_files_shuffled_:
                    bkgd_label_file = bkgd_file[:-3] + 'png'
                    bkgd_img = cv2.resize(cv2.imread(self.folder_learn_iter + '/query_added/images/' + bkgd_file), self.img_size)
                    bkgd_label = cv2.resize(cv2.imread(self.folder_learn_iter + '/query_added/labels/' + bkgd_label_file, cv2.IMREAD_GRAYSCALE), self.img_size)
                    
                    img_bkgd = g_utils.find_inpaint_rotation(img_1 = img, label_1 = label, img_2 = bkgd_img, label_2 = bkgd_label)
                    if img_bkgd.any() != 0:
                        img_new = self.img_inpaint(img, label, tool_extract_map, bkgd = img_bkgd, fusion_type = 'avg', fusion_blur = 15, img_size = self.img_size)
                        cv2.imwrite(self.folder_learn_iter + '/background/' + img_file, img_new)
                        break
                if img_bkgd.all() == 0:
                    continue       
                        
                
            
        
        return None
    
    def generate_syn_img(self, image, label, dilation = 60, syn1 = 10, syn2 = 10, multi_gen = 2, multi_gen_same = False, flip=0.5, shrink_factor=0.9, 
                         x_factor=0.1, y_factor=0.1, r_factor=5.0, 
                         color_adjust_strength=1.0, brightness_adjust=1.2, border_fusion = True, fusion_blur=20, 
                         border_center=(120,120), radius=115, ksize=(2,2), sig_x=2,
                         elastic=(-2, -1), els_alpha=2500, els_sigma=12, 
                         dila_ero=0, val_type=0, dila_ero_size=1, 
                         img_elastic=-1, img_els_alpha=2500, img_els_sigma=12,
                         img_size = (240,240), show_img = False):
        
        self.input_img = np.copy(image)
        self.input_tool, self.input_label, self.input_extract_map = self.extract_tool(image, label, dilation = self.extract_dilation)
        
        syn1_images_ = []
        syn1_labels_ = []
        syn2_images_ = []
        syn2_labels_ = []

        if syn1 < 1:
            if rand_from_range(0,1) < syn1:
                syn1 = 1
            else:
                syn1 = 0
                
        if syn2 < 1:
            if rand_from_range(0,1) < syn2:
                syn2 = 1
            else:
                syn2 = 0
        
        for iteration in range(0, syn1):
            self.syn_type = 'syn1_' + str(iteration)
            self.select_tool(random_tool = False, 
                             flip = rand_from_range(flip[0], flip[1]),
                             img_size = img_size,
                             show_img = show_img)
            self.select_background(random_bkgd = True, img_size = img_size, show_img = show_img)
            
            self.shrink_tool(factor = rand_from_range(shrink_factor[0], shrink_factor[1]), show_img = show_img)
            
            self.move_tool(x_factor = rand_from_range(x_factor[0], x_factor[1]), 
                           y_factor = rand_from_range(y_factor[0], y_factor[1]), 
                           r_factor =rand_from_range(r_factor[0], r_factor[1]),
                           show_img = show_img)
            
            for iteration in range(0, multi_gen):
                self.syn_type = 'syn1_' + str(iteration) + '_multi_' + str(iteration)
                if (iteration%2) == 0:
                    fusion_type = 'avg'
                    # border_fusion = True
                else:
                    fusion_type = 'gauss'
                    # border_fusion = True
                self.render_reflection(index = None,
                                        refl_type = 2, 
                                        refl_param = 1,
                                        use_diverse_refl = 1, 
                                        diverse_bkgd_path = 'val2017',
                                        show_img = show_img)
                
                self.combine_syn_img(color_adjust_strength = rand_from_range(color_adjust_strength[0], color_adjust_strength[1]), 
                                        brightness_adjust = rand_from_range(brightness_adjust[0], brightness_adjust[1]),
                                        border_fusion = border_fusion,
                                        fusion_blur = int(rand_from_range(fusion_blur[0], fusion_blur[1])),
                                        fusion_type = fusion_type,
                                        show_img = show_img)
                
                self.render_shadow(rander_shadow = -1, 
                                               tool_shadow_range = 1, 
                                               tool_shadow_brightness = 1, 
                                               bkgd_shadow_range = 1, 
                                               bkgd_shadow_brightness = 1,
                                               show_img = show_img)
                
                self.render_exposure(render_exposure = -1,
                                                 expo_lenth = 1,
                                                 expo_width = 1,
                                                 expo_taper = 1,
                                                 expo_strength = 1, 
                                                 expo_spread = 1,
                                                 expo_angle = 1, 
                                                 expo_move = 1,
                                                 show_img = show_img)
                
                self.set_bound_and_blur(border_type = 'rect', border_center = (rand_from_range(border_center[0][0], border_center[0][1]), rand_from_range(border_center[1][0], border_center[1][1])),
                                        radius = rand_from_range(radius[0], radius[1]), 
                                        border_rect = (rand_from_range(6, 9),rand_from_range(6, 9),rand_from_range(71, 74),rand_from_range(71, 74)),
                                        border_color = [[18.9, 5.5], [17.7, 5.3], [24.6, 7.1]],
                                        ksize = rand_from_range(ksize[0], ksize[1]),
                                        sig_x = rand_from_range(sig_x[0], sig_x[1]),
                                        show_img = show_img)
                
                self.elastic_dilation_erosion(elastic = rand_from_range(elastic[0], elastic[1]), 
                                                els_alpha = rand_from_range(els_alpha[0], els_alpha[1]), 
                                                els_sigma = rand_from_range(els_sigma[0], els_sigma[1]), 
                                                dila_ero = rand_from_range(dila_ero[0], dila_ero[1]), 
                                                val_type = 0, 
                                                dila_ero_size = rand_from_range(dila_ero_size[0], dila_ero_size[1]),
                                                img_elastic = rand_from_range(img_elastic[0], img_elastic[1]), 
                                                img_els_alpha = rand_from_range(img_els_alpha[0], img_els_alpha[1]), 
                                                img_els_sigma = rand_from_range(img_els_sigma[0], img_els_sigma[1]))
                
                syn1_image = self.syn_img_final['img']
                syn1_label = self.syn_img_final['tool_map']
                syn1_images_.append(syn1_image)
                syn1_labels_.append(syn1_label)
                if multi_gen_same == True:
                    for i in range(0, multi_gen - 1):
                        syn1_images_.append(syn1_image)
                        syn1_labels_.append(syn1_label)
                    break
                
        syn2_counter = 0
        syn2_attempting = 0
        syn2_attempting_max = syn2*10
        while syn2_counter < syn2 and syn2_attempting < syn2_attempting_max:
            self.syn_type = 'syn2_' + str(syn2_counter)
            syn2_attempting = syn2_attempting + 1
            
            self.select_tool(random_tool = True, 
                             flip = rand_from_range(flip[0], flip[1]),
                             img_size = img_size,
                             show_img = show_img)
            self.select_background(random_bkgd = False, img_size = img_size, show_img = show_img)
            
            self.shrink_tool(factor = rand_from_range(shrink_factor[0], shrink_factor[1]), show_img = show_img)
            
            self.move_tool(x_factor = rand_from_range(x_factor[0], x_factor[1]), 
                           y_factor = rand_from_range(y_factor[0], y_factor[1]), 
                           r_factor =rand_from_range(r_factor[0], r_factor[1]),
                           show_img = show_img)
            
            
            for iteration in range(0, multi_gen):
                self.syn_type = 'syn2_' + str(syn2_counter) + '_multi_' + str(iteration)
                self.render_reflection(index = None,
                                        refl_type = 2, 
                                        refl_param = 1,
                                        use_diverse_refl = 1, 
                                        diverse_bkgd_path = 'val2017',
                                        show_img = show_img)
                
                if (iteration%2) == 0:
                    fusion_type = 'avg'
                    # border_fusion = True
                else:
                    fusion_type = 'gauss'
                    # border_fusion = True
                self.combine_syn_img(color_adjust_strength = rand_from_range(color_adjust_strength[0], color_adjust_strength[1]), 
                                        brightness_adjust = rand_from_range(brightness_adjust[0], brightness_adjust[1]),
                                        border_fusion = border_fusion,
                                        fusion_blur = int(rand_from_range(fusion_blur[0], fusion_blur[1])),
                                        fusion_type = fusion_type,
                                        show_img = show_img)
                
                self.render_shadow(rander_shadow = -1, 
                                               tool_shadow_range = 1, 
                                               tool_shadow_brightness = 1, 
                                               bkgd_shadow_range = 1, 
                                               bkgd_shadow_brightness = 1,
                                               show_img = show_img)
                
                self.render_exposure(render_exposure = -1,
                                                 expo_lenth = 1,
                                                 expo_width = 1,
                                                 expo_taper = 1,
                                                 expo_strength = 1, 
                                                 expo_spread = 1,
                                                 expo_angle = 1, 
                                                 expo_move = 1,
                                                 show_img = show_img)
                
                self.set_bound_and_blur(border_type = 'rect', border_center = (rand_from_range(border_center[0][0], border_center[0][1]), rand_from_range(border_center[1][0], border_center[1][1])),
                                        radius = rand_from_range(radius[0], radius[1]), 
                                        border_rect = (rand_from_range(6, 9),rand_from_range(6, 9),rand_from_range(71, 74),rand_from_range(71, 74)),
                                        border_color = [[18.9, 5.5], [17.7, 5.3], [24.6, 7.1]],
                                        ksize = rand_from_range(ksize[0], ksize[1]),
                                        sig_x = rand_from_range(sig_x[0], sig_x[1]),
                                        show_img = show_img)
                
                self.elastic_dilation_erosion(elastic = rand_from_range(elastic[0], elastic[1]), 
                                                els_alpha = rand_from_range(els_alpha[0], els_alpha[1]), 
                                                els_sigma = rand_from_range(els_sigma[0], els_sigma[1]), 
                                                dila_ero = rand_from_range(dila_ero[0], dila_ero[1]), 
                                                val_type = 0, 
                                                dila_ero_size = rand_from_range(dila_ero_size[0], dila_ero_size[1]),
                                                img_elastic = rand_from_range(img_elastic[0], img_elastic[1]), 
                                                img_els_alpha = rand_from_range(img_els_alpha[0], img_els_alpha[1]), 
                                                img_els_sigma = rand_from_range(img_els_sigma[0], img_els_sigma[1]))
                
                syn2_image = self.syn_img_final['img']
                syn2_label = self.syn_img_final['tool_map']
                syn2_images_.append(syn2_image)
                syn2_labels_.append(syn2_label)
                if multi_gen_same == True:
                    for i in range(0, multi_gen - 1):
                        syn1_images_.append(syn1_image)
                        syn1_labels_.append(syn1_label)
                    break
            syn2_counter = syn2_counter + 1
            
        return syn1_images_, syn1_labels_, syn2_images_, syn2_labels_
    
    def extract_tool(self, image, label, dilation = 60):
        tool_label = label
        tool_mask = tool_label[:]
        tool_mask_small = g_utils.img_dilatation(np.copy(tool_mask), 0, int(dilation/2))
        tool_mask_large = g_utils.img_dilatation(np.copy(tool_mask), 0, dilation)
        # tool_label = tool_label.clip(0, 1)
        tool_mask_small[tool_mask_small>0] = 1
        tool_mask_large[tool_mask_large>0] = 1
        
        
        tool_img = image[:]
        # for chn in range(0,3):
        #     tool_img[:,:,chn] = image[:,:,chn] * tool_mask_large
        tool_extract_map = tool_mask_small
        return tool_img, tool_label, tool_extract_map
    
    def extract_tool_from_folder(self, source_folder, dilation = 60):
        idx_tool = 0
        
        img_folder = source_folder + '/images'
        label_folder = source_folder + '/labels'
        
        img_files_ = al.get_file_names(img_folder, 'jpg')
        for img_file in img_files_:

            label_file = img_file[:-3] + 'png'

            tool_label = cv2.imread(label_folder + '/' + label_file)
            tool_mask = tool_label[:]
            tool_mask_small = g_utils.img_dilatation(np.copy(tool_mask), 0, int(dilation/2))
            tool_mask_large = g_utils.img_dilatation(np.copy(tool_mask), 0, dilation)
            # tool_label = tool_label.clip(0, 1)
            tool_mask_small[tool_mask_small>0] = 1
            tool_mask_large[tool_mask_large>0] = 1
            tool_img = cv2.imread(img_folder + '/' + img_file) #* tool_mask_large
            
            if np.max(tool_img) <= 20:
                continue
            
            tool_img_folder = self.folder_tool[self.cur_iter] + '/tool'
            tool_map_folder = self.folder_tool[self.cur_iter] + '/tool_map'
            tool_extract_map_folder = self.folder_tool[self.cur_iter] + '/tool_extract_map'
            al.create_folder(tool_img_folder)
            al.create_folder(tool_map_folder)
            al.create_folder(tool_extract_map_folder)
            
            cv2.imwrite(tool_img_folder + '/tool_' + str(idx_tool) + '.jpg',  tool_img)
            cv2.imwrite(tool_map_folder + '/tool_' + str(idx_tool) + '.png',  tool_label)
            cv2.imwrite(tool_extract_map_folder + '/tool_' + str(idx_tool) + '.png',  tool_mask_small)
            
            idx_tool = idx_tool + 1
            # cv2.imwrite(self.folder_tool + '/tool_' + str(self.idx_tool) + '.bmp',  tool_img)
            # cv2.imwrite(self.folder_tool + '/tool_map_' + str(self.idx_tool) + '.bmp',  tool_label)
            # cv2.imwrite(self.folder_tool + '/tool_extract_mask_' + str(self.idx_tool) + '.bmp',  tool_mask_small)
        return None
    
    def load_tool_and_bkgd(self):
        self.list_tool_ = al.get_file_names(self.folder_tool[self.cur_iter] + '/tool', 'jpg')
        self.list_bkgd_ = al.get_file_names(self.folder_bkgd[self.cur_iter], 'jpg')
        return None
    
    def select_tool(self, random_tool = True, flip = 0.5, img_size = (240,240), show_img = False):
        
        if random_tool == True:
            tool_file = random.choice(self.list_tool_)
            tool_map_file = tool_file[:-3] + 'png'
            folder_tool_cur = self.folder_tool[self.cur_iter]
                
            self.tool['img'] = cv2.imread(folder_tool_cur + '/tool/' + tool_file)
            tool_map = cv2.imread(folder_tool_cur + '/tool_map/' + tool_map_file, cv2.IMREAD_GRAYSCALE)
            tool_extract_mask = cv2.imread(folder_tool_cur + '/tool_extract_map/' + tool_map_file, cv2.IMREAD_GRAYSCALE)
            
            self.tool['img'] = cv2.resize(self.tool['img'], img_size)
            tool_map = cv2.resize(tool_map, img_size)
            tool_extract_mask = cv2.resize(tool_extract_mask, img_size)


            self.tool['tool_map'] = np.zeros((tool_map.shape[0], tool_map.shape[1],2))
            self.tool['tool_map'][:,:,0] = tool_map
            self.tool['tool_map'][:,:,1] = tool_extract_mask
              
            if flip < 0:
                self.tool['img'] = cv2.flip(self.tool['img'], 1)
                self.tool['tool_map'] = cv2.flip(self.tool['tool_map'], 1)
                # self.tool['distance_map'] = cv2.flip(self.tool['distance_map'].astype(np.int16), 0)
        else:
            self.tool['img'] = self.input_img[:]
            tool_map = self.input_label
            tool_extract_mask = self.input_extract_map

            self.tool['img'] = cv2.resize(self.tool['img'], img_size)
            tool_map = cv2.resize(tool_map, img_size)
            tool_extract_mask = cv2.resize(tool_extract_mask, img_size)
            
            self.tool['tool_map'] = np.zeros((tool_map.shape[0], tool_map.shape[1],2))
            self.tool['tool_map'][:,:,0] = tool_map
            self.tool['tool_map'][:,:,1] = tool_extract_mask
            
            self.tool['img'] = cv2.resize(self.tool['img'], img_size)
            tool_map = cv2.resize(tool_map, img_size)
              
            if flip < 0:
                self.tool['img'] = cv2.flip(self.tool['img'], 1)
                self.tool['tool_map'] = cv2.flip(self.tool['tool_map'], 1)
            
        if show_img == True:
            tool_img = self.tool['img']
            # tool_img[tool_img==0] = 255
            g_utils.show_img('selected_tool', tool_img)
            cv2.imwrite('temp_img/1_selected_tool.jpg', self.tool['img'])
        if self.output_process == True:
            cv2.imwrite(self.folder_learn_iter + '/query_added/images/' + self.file_name + '_' + self.syn_type + '_' + '1_selected_tool.jpg', self.tool['img'])
        return None
    
    def select_background(self, random_bkgd = True, img_size = (240, 240), show_img = False):
        
        if random_bkgd == True:
            bkgd_file = random.choice(self.list_bkgd_)
            
            img_bkgd = cv2.resize(cv2.imread(self.folder_bkgd[self.cur_iter] + '/' + bkgd_file), img_size)
            self.background['img'] = np.copy(img_bkgd)
            self.random_bkgd = True
        else:
            self.random_bkgd = False
            img = np.copy(self.input_img[:])
            label = self.input_label
            tool_extract_map = self.input_extract_map
            
            img_bkgd = g_utils.find_inpaint_rotation(img_1 = img, label_1 = label, img_2 = img, label_2 = label)           
            if img_bkgd.all() == 0 or g_utils.rand_from_range(0, 1)<0.5:
                bkgd_file = random.choice(self.list_bkgd_)    
                img_bkgd = cv2.resize(cv2.imread(self.folder_bkgd[self.cur_iter] + '/' + bkgd_file), img_size)
            else:
                img_bkgd = img_bkgd
                 
            img_new = self.img_inpaint(img, label, tool_extract_map, bkgd = img_bkgd, fusion_type = 'avg', fusion_blur = 15, img_size = self.img_size)
        
            self.background['img'] = img_new
        
        if show_img == True:
            g_utils.show_img('selected_bkgd', self.background['img'])
            cv2.imwrite('temp_img/2_selected_bkgd.jpg', self.background['img'])
        if self.output_process == True:
            cv2.imwrite(self.folder_learn_iter + '/query_added/images/' + self.file_name + '_' + self.syn_type + '_' + '2_selected_bkgd.jpg', self.background['img'])
            a=1
        return None
    
    def shrink_tool(self, factor, show_img = False):
        if factor > 0.99 and factor < 1.0:
            factor = 0.95
        if factor >= 1.0 and factor < 1.01:
            factor = 1.05
            
        if factor < 1:
            tool_shape_org = self.tool['img'].shape
            
            # shape_sk = (g_utils.round_up_to_odd(factor * tool_shape_org[0]), g_utils.round_up_to_odd(factor * tool_shape_org[1]))
            size_n = (int(factor*self.img_size[0]), int(factor*self.img_size[1]))
            shape_sk = size_n
            # factor = shape_sk/tool_shape_org[0]
    
            tool_shrinked = cv2.resize(self.tool['img'], size_n)
            img_new = np.zeros(self.tool['img'].shape, np.uint8)
            # img_new[int(0.5*(tool_shape_org[1] - shape_sk[1])) : int(0.5*(tool_shape_org[1] + shape_sk[1])), int(0.5*(tool_shape_org[0] - shape_sk[0])) : int(0.5*(tool_shape_org[0] + shape_sk[0])) , :] = tool_shrinked
            img_new = g_utils.center_add_image(small_img = tool_shrinked, large_img = img_new)
            
            tool_map_int = self.tool['tool_map'].astype(np.uint8)
            tool_map_shrinked = cv2.resize(tool_map_int, size_n)
            tool_map_new = np.zeros(self.tool['tool_map'].shape, np.uint8)
            # tool_map_new[int(0.5*(tool_shape_org[1] - shape_sk[1])) : int(0.5*(tool_shape_org[1] + shape_sk[1])), int(0.5*(tool_shape_org[0] - shape_sk[0])) : int(0.5*(tool_shape_org[0] + shape_sk[0])) , :] = tool_map_shrinked
            tool_map_new = g_utils.center_add_image(small_img = tool_map_shrinked, large_img = tool_map_new)
            
            # movement = int(0.5*(tool_shape_org[0] - shape_sk[0]))
            movement = 0
        
        if factor > 1:
            size_n = (int(factor*self.img_size[0]), int(factor*self.img_size[1]))
            img_new = cv2.resize(np.copy(self.tool['img']), size_n)
            tool_map_new = cv2.resize(self.tool['tool_map'], size_n)     
            shape_sk = size_n
       
        self.tool_shrinked = pd.Series({'index': -1,
                                        'img': img_new,
                                        'type': [self.tool['type'], self.tool['index']],
                                        'contour': self.tool['contour'],
                                        'tool_map': tool_map_new,
                                        'lines': self.tool['lines'],
                                        'distance_map': self.tool['distance_map'],
                                        'geo_feature_points': self.tool['geo_feature_points'],
                                        'geo_feature_vectors': self.tool['geo_feature_vectors'],
                                        'render_feature_points': self.tool['render_feature_points'],
                                        'render_feature_vectors': self.tool['render_feature_vectors'],
                                        'shrink_factors': {'factor': factor, 'shape_shrinked':shape_sk, 'dismap_shape_shrinked' : None} })
        if show_img == True:
            img_new[img_new==0] = 255
            g_utils.show_img('tool_shrinked',img_new)
            cv2.imwrite('temp_img/3_tool_shrinked.jpg', img_new)
        if self.output_process == True:
            cv2.imwrite(self.folder_learn_iter + '/query_added/images/' + self.file_name + '_' + self.syn_type + '_' + '3_tool_shrinked.jpg', img_new)
            a=1
            
        return None
    
    def move_tool(self, x_factor, y_factor, r_factor, show_img = False):
        # first, if the tool is moved out of the image, set the image to be zeros
        img_shape = self.tool_shrinked['img'].shape
        # print(img_shape)
        shrinked_shape = self.tool_shrinked['shrink_factors']['shape_shrinked']
        x_movement = int( x_factor * shrinked_shape[0] )
        y_movement = int( y_factor * shrinked_shape[1] )
        
        T_mat = np.array([[1.0, 0.0, y_movement + 0.5*(img_shape[0] - shrinked_shape[1])],
                          [0.0, 1.0, x_movement]])
        
        R_mat = cv2.getRotationMatrix2D((0.5*img_shape[0], 0.5*img_shape[1]), r_factor, scale=1) # these T matrix and R matrix are for opencv, not for translating and rotating points and vectors, they are defined later
        
        if  y_movement >= (0.5*( img_shape[0] + shrinked_shape[1])) or abs(x_movement) >= img_shape[0]:
            img_new = np.zeros(self.tool_shrinked['img'].shape, np.uint8)
            tool_map_new = np.zeros(self.tool_shrinked['tool_map'].shape, np.uint8)
        else:
            # translate and rotate the image
            img_new = cv2.warpAffine(self.tool_shrinked['img'], T_mat, (img_shape[1], img_shape[0])) 
            img_new = cv2.warpAffine(img_new, R_mat, (img_shape[1], img_shape[0])) 
            # img_new = g_utils.center_add_image(small_img = img_new, large_img = np.zeros((img_shape[1], img_shape[0], 3)))
            # translate and rotate the tool_map
            tool_map_new = cv2.warpAffine(self.tool_shrinked['tool_map'], T_mat, (img_shape[1], img_shape[0])) 
            tool_map_new = cv2.warpAffine(tool_map_new, R_mat, (img_shape[1], img_shape[0])) 
            # tool_map_new = g_utils.center_add_image(small_img = tool_map_new, large_img = np.zeros((img_shape[1], img_shape[0], 2)))
            
        # print(img_new.shape)
        # print(tool_map_new.shape)
        
        s_factor = self.tool_shrinked['shrink_factors']['factor']
        if  s_factor > 1:
            img_org = np.copy(img_new)
            tool_map_org = np.copy(tool_map_new)
            
            width_l = int(s_factor*self.img_size[0])
            height_l = int(s_factor*self.img_size[1])
            width_s = self.img_size[0]
            height_s = self.img_size[1]
            
            # center
            img_new = img_org[int(0.5*(height_l - height_s)): int(0.5*(height_l + height_s)), int(0.5*(width_l - width_s)): int(0.5*(width_l + width_s)), :]
            tool_map_new = tool_map_org[int(0.5*(height_l - height_s)): int(0.5*(height_l + height_s)), int(0.5*(width_l - width_s)): int(0.5*(width_l + width_s)), :]
            
            # top left
            img_temp = img_org[0:height_s,0:width_s,:]
            tool_map_temp = tool_map_org[0:height_s,0:width_s,:]
            if np.sum(tool_map_temp[:,:,0]) > np.sum(tool_map_new[:,:,0]):
                img_new = img_temp
                tool_map_new = tool_map_temp 
            # top right
            img_temp = img_org[0:height_s,-width_s:,:]
            tool_map_temp = tool_map_org[0:height_s,-width_s:,:]
            if np.sum(tool_map_temp[:,:,0]) > np.sum(tool_map_new[:,:,0]):
                img_new = img_temp
                tool_map_new = tool_map_temp 
            # bot left
            img_temp = img_org[-height_s:,0:width_s,:]
            tool_map_temp = tool_map_org[-height_s:,0:width_s,:]
            if np.sum(tool_map_temp[:,:,0]) > np.sum(tool_map_new[:,:,0]):
                img_new = img_temp
                tool_map_new = tool_map_temp 
            # bot right
            img_temp = img_org[-height_s:,-width_s:,:]
            tool_map_temp = tool_map_org[-height_s:,-width_s:,:]
            if np.sum(tool_map_temp[:,:,0]) > np.sum(tool_map_new[:,:,0]):
                img_new = img_temp
                tool_map_new = tool_map_temp 
            
        
        
        
        self.tool_moved = pd.Series({'index': self.tool_shrinked['index'],
                                        'img': img_new,
                                        'type': self.tool_shrinked['type'],
                                        'contour': self.tool_shrinked['contour'],
                                        'tool_map': tool_map_new,
                                        'lines': self.tool_shrinked['lines'],
                                        'distance_map': self.tool_shrinked['distance_map'],
                                        'geo_feature_points': self.tool_shrinked['geo_feature_points'],
                                        'geo_feature_vectors': self.tool_shrinked['geo_feature_vectors'],
                                        'render_feature_points': self.tool_shrinked['render_feature_points'],
                                        'render_feature_vectors': self.tool_shrinked['render_feature_vectors'],
                                        'shrink_factors': self.tool_shrinked['shrink_factors'],
                                        'move_factors':{'x_movement': x_movement, 'y_movement': y_movement + 0.5*(1001-shrinked_shape[1]), 'rotate_degree': r_factor}})

        # show some results for debugging----------------------------
        # g_utils.show_contour_distance_map('test_dismap1', self.tool_shrinked['distance_map'], self.tool_shrinked['contour'])
        # g_utils.show_contour_distance_map('test_dismap2', distance_map_new, contour_new)
        # img_temp = g_utils.draw_vectors(img_new, [render_feature_points_new[1], render_feature_points_new[3], render_feature_points_new[2]], render_feature_vectors_new, color = (100,200,0))
        # g_utils.show_img('12',img_temp)
        # temp_img = g_utils.draw_points(img_new , render_feature_points_new)
        if show_img ==True:
            img_new[img_new==0] = 255
            g_utils.show_img('tool_moved', img_new)
            cv2.imwrite('temp_img/4_tool_moved.jpg', img_new)
        if self.output_process == True:    
            cv2.imwrite(self.folder_learn_iter + '/query_added/images/' + self.file_name + '_' + self.syn_type + '_' + '4_tool_moved.jpg', img_new)
            a=1
        return None
    
    # [function] render_reflection
    # [Discription]: render the reflection of the tool, a background from the same group of backgrounds will be chosen and render the metal texture of the tool
    # [parameters]: refl_type - if 0, there will be no reflection rendered. refl_param - a list of numbers wokring with reflection type, if None is given, defult parameters will be used
    # [result]: a pd.series contains rendered image and the index of the relfection background
    # [return]: None
    def render_reflection(self, index = None, refl_type = 1, refl_param = None, use_diverse_refl = -0.5, diverse_bkgd_path = 'val2017', show_img = False):
        
        self.tool_refl = pd.Series({'index': self.tool_moved['index'],
                                    'img': self.tool_moved['img'],
                                    'type': self.tool_moved['type'],
                                    'contour': self.tool_moved['contour'],
                                    'tool_map': self.tool_moved['tool_map'],
                                    'lines': self.tool_moved['lines'],
                                    'distance_map': self.tool_moved['distance_map'],
                                    'geo_feature_points': self.tool_moved['geo_feature_points'],
                                    'geo_feature_vectors': self.tool_moved['geo_feature_vectors'],
                                    'render_feature_points': self.tool_moved['render_feature_points'],
                                    'render_feature_vectors': self.tool_moved['render_feature_vectors'],
                                    'shrink_factors': self.tool_moved['shrink_factors'],
                                    'move_factors':self.tool_moved['move_factors'],
                                    'refl_factors': [None, None, [None, None]]})
        
        if show_img == True:
            g_utils.show_img('tool_refl', self.tool_moved['img'])
            cv2.imwrite('temp_img/5_tool_refl.jpg', self.tool_moved['img'])
        if self.output_process == True:
            a=1
        
        return None
    
    # [function] combine_syn_img
    # [Discription]: combine the tool image and the background, adjusting the color style of the tool to match the color style of the background
    # [parameters]: color_adjust_factor - if 1 the tool's color will be adjust to the same BGR ratio as the background, if 0, the tool's color will not be adjusted
    # [result]: a pd.series containing the combined synthetic image and relative information including 
    # [return]: None
    def combine_syn_img(self, color_adjust_strength = 0.8, brightness_adjust = 1, border_fusion = False, fusion_blur = 20, fusion_type = 'avg', show_img = False): 
        
        # first, adjust the color style of the tool
        img_bkgd = self.background['img'][:].astype(np.float)
        color_style_bkgd = np.array([np.mean(img_bkgd[:,:,0]), np.mean(img_bkgd[:,:,1]), np.mean(img_bkgd[:,:,2])])
        
        img_tool = self.tool_refl['img'][:].astype(np.float)
        
        img_tool_chn0 = img_tool[:,:,0]
        img_tool_chn1 = img_tool[:,:,1]
        img_tool_chn2 = img_tool[:,:,2]
        color_style_tool = np.array([np.mean(img_tool_chn0[img_tool_chn0!=0]), np.mean(img_tool_chn1[img_tool_chn1!=0]), np.mean(img_tool_chn2[img_tool_chn2!=0])])
        
        color_adjust_factor = color_style_bkgd / color_style_tool
        
        
        img_tool_adjusted = np.zeros(img_tool.shape, np.float)
        for chn in range(0,3):
            img_tool_adjusted[:,:,chn] = img_tool[:,:,chn] * color_adjust_factor[chn]
          
        img_tool_adjusted = ((color_adjust_strength * img_tool_adjusted + (1 - color_adjust_strength) * img_tool)).astype(np.float).clip(0,255)
        
        bright_bkgd = np.mean(img_bkgd)
        bright_tool = np.mean(img_tool_adjusted[img_tool_adjusted!=0])
        
        # print(bright_bkgd / bright_tool)
        img_tool_adjusted = (img_tool_adjusted * brightness_adjust * bright_bkgd / bright_tool).clip(0,255)
        
        tool_map = self.tool_refl['tool_map'][:,:,0]
        tool_extract_mask = self.tool_refl['tool_map'][:,:,1]
        
        if border_fusion == True:
            # img_tool_adjusted[img_tool_adjusted == 0] = 200
            
            tool_mask_blur = tool_extract_mask.astype(np.float32)  #+ tool_map
            tool_mask_blur = tool_mask_blur * 100
            # tool_mask_blur = cv2.GaussianBlur(tool_mask_blur, (15,15), 0)
            if fusion_type == 'avg':
                tool_mask_blur = cv2.blur(tool_mask_blur, (fusion_blur, fusion_blur))
            if fusion_type == 'gauss':
                fusion_blur = g_utils.round_up_to_odd(fusion_blur)
                tool_mask_blur = cv2.GaussianBlur(tool_mask_blur, (fusion_blur,fusion_blur), fusion_blur)
            tool_extract_mask = tool_mask_blur/100
        
        tool_map_reverse = np.ones(tool_extract_mask.shape, np.float32) - tool_extract_mask
        
        for chn in range(0,3):
            img_tool_adjusted[:,:,chn] = img_tool_adjusted[:,:,chn] * tool_extract_mask
            img_bkgd[:,:,chn] = img_bkgd[:,:,chn] * tool_map_reverse
            
        img_combined = (img_bkgd + img_tool_adjusted).clip(0,255).astype(np.uint8)
        
        self.syn_img_comb = pd.Series({'index': self.tool_refl['index'],
                                        'img': img_combined,
                                        'type': self.tool_refl['type'],
                                        'contour': self.tool_refl['contour'],
                                        'tool_map': tool_map,
                                        'lines': self.tool_refl['lines'],
                                        'distance_map': self.tool_refl['distance_map'],
                                        'geo_feature_points': self.tool_refl['geo_feature_points'],
                                        'geo_feature_vectors': self.tool_refl['geo_feature_vectors'],
                                        'render_feature_points': self.tool_refl['render_feature_points'],
                                        'render_feature_vectors': self.tool_refl['render_feature_vectors'],
                                        'shrink_factors': self.tool_refl['shrink_factors'],
                                        'move_factors':self.tool_refl['move_factors'],
                                        'refl_factors': self.tool_refl['refl_factors'],
                                        'bkgd_idx': self.background['index'],
                                        'bkgd_img': self.background['img'],
                                        'comb_factors': [color_style_tool, color_style_bkgd, color_adjust_strength]})
        if show_img == True:
            g_utils.show_img('img_combined', img_combined)
            g_utils.show_img('tool_mask_blur', tool_mask_blur.astype(np.uint8))
            cv2.imwrite('temp_img/6_img_combined.jpg', img_combined)   
            cv2.imwrite('temp_img/6_1_img_combined_bkgd.jpg', img_bkgd)
        if self.output_process == True:

            
            tool_map_v = g_utils.visual_mask(tool_map, [0, 250, 0])
            tool_map_v = g_utils.make_transparent_bkgd(tool_map_v)
            # tool_map_v = (tool_map_v + img_combined).clip(0,255)
            
            tool_extract_mask_v = g_utils.visual_mask(tool_extract_mask, [255, 0, 0])
            # tool_extract_mask_v = g_utils.make_transparent_bkgd(tool_extract_mask_v)
            # tool_extract_mask_v = (tool_extract_mask_v + img_combined).clip(0,255)
            
            tool_extract_mask_reverse_v = g_utils.visual_mask(tool_map_reverse, [0, 255, 255])
            # tool_extract_mask_reverse_v = g_utils.make_transparent_bkgd(tool_extract_mask_reverse_v)
            
            tool_extract_mask_combined_v = (tool_extract_mask_v + tool_extract_mask_reverse_v).clip(0, 255)

            cv2.imwrite(self.folder_learn_iter + '/query_added/images/' + self.file_name + '_' + self.syn_type + '_' + '6_img_combined.jpg', img_combined)
            cv2.imwrite(self.folder_learn_iter + '/query_added/images/' + self.file_name + '_' + self.syn_type + '_' + '6_tool_map_.png', tool_map_v)
            cv2.imwrite(self.folder_learn_iter + '/query_added/images/' + self.file_name + '_' + self.syn_type + '_' + '6_tool_extract_mask.png', tool_extract_mask_v)
            cv2.imwrite(self.folder_learn_iter + '/query_added/images/' + self.file_name + '_' + self.syn_type + '_' + '6_tool_extract_mask_reverse.png', tool_extract_mask_reverse_v)
            cv2.imwrite(self.folder_learn_iter + '/query_added/images/' + self.file_name + '_' + self.syn_type + '_' + '6_tool_extract_mask_combined.png', tool_extract_mask_combined_v)
        return None
    
    
    # [function] render_shadow
    # [Discription]: render shadow on both tool and background
    # [parameters]: range - pixels that have distance larger than the range will not have rendered shadow, 
    # [result]: a pd.series containing the combined synthetic image and relative information including 
    # [return]: None
    def render_shadow(self, rander_shadow = 0.5, tool_shadow_range = 100, tool_shadow_brightness = 0.5, bkgd_shadow_range = 20, bkgd_shadow_brightness = 0.1, show_img = False):
        img_new = self.syn_img_comb['img']
        self.syn_img_shad = pd.Series({'index': self.syn_img_comb['index'],
                                        'img': img_new,
                                        'type': self.syn_img_comb['type'],
                                        'contour': self.syn_img_comb['contour'],
                                        'tool_map': self.syn_img_comb['tool_map'],
                                        'lines': self.syn_img_comb['lines'],
                                        'distance_map': self.syn_img_comb['distance_map'],
                                        'geo_feature_points': self.syn_img_comb['geo_feature_points'],
                                        'geo_feature_vectors': self.syn_img_comb['geo_feature_vectors'],
                                        'render_feature_points': self.syn_img_comb['render_feature_points'],
                                        'render_feature_vectors': self.syn_img_comb['render_feature_vectors'],
                                        'shrink_factors': self.syn_img_comb['shrink_factors'],
                                        'move_factors':self.syn_img_comb['move_factors'],
                                        'refl_factors': self.syn_img_comb['refl_factors'],
                                        'bkgd_idx': self.syn_img_comb['bkgd_idx'],
                                        'bkgd_img': self.syn_img_comb['bkgd_img'],
                                        'comb_factors': self.syn_img_comb['comb_factors'],
                                        'shadow_factors': [tool_shadow_range, tool_shadow_brightness, bkgd_shadow_range, bkgd_shadow_brightness]})
        
        if show_img == True:
            g_utils.show_img('img_shadowed', img_new)
            cv2.imwrite('temp_img/7_img_shadowed.jpg', img_new)
        return None
    
    def render_exposure(self, render_exposure, expo_lenth ,expo_width, expo_taper, expo_strength = 5, expo_spread = 51, expo_angle = 0, expo_move = [100, 0], show_img = False):
        img_expo = self.syn_img_shad['img']
        self.syn_img_expo = pd.Series({'index': self.syn_img_shad['index'],
                                        'img': img_expo,
                                        'type': self.syn_img_shad['type'],
                                        'contour': self.syn_img_shad['contour'],
                                        'tool_map': self.syn_img_shad['tool_map'],
                                        'lines': self.syn_img_shad['lines'],
                                        'distance_map': self.syn_img_shad['distance_map'],
                                        'geo_feature_points': self.syn_img_shad['geo_feature_points'],
                                        'geo_feature_vectors': self.syn_img_shad['geo_feature_vectors'],
                                        'render_feature_points': self.syn_img_shad['render_feature_points'],
                                        'render_feature_vectors': self.syn_img_shad['render_feature_vectors'],
                                        'shrink_factors': self.syn_img_shad['shrink_factors'],
                                        'move_factors':self.syn_img_shad['move_factors'],
                                        'refl_factors': self.syn_img_shad['refl_factors'],
                                        'bkgd_idx': self.syn_img_shad['bkgd_idx'],
                                        'bkgd_img': self.syn_img_shad['bkgd_img'],
                                        'comb_factors': self.syn_img_shad['comb_factors'],
                                        'shadow_factors': self.syn_img_shad['shadow_factors'],
                                        'expo_factors': {'expo_lenth':expo_lenth,
                                                         'expo_width':expo_width,
                                                         'expo_taper':expo_taper,
                                                         'expo_strength': expo_strength,
                                                         'expo_angle': expo_angle,
                                                         'expo_move': expo_move}})
        if show_img == True:
            # g_utils.show_img('expo_map', expo_map_show)
            g_utils.show_img('img_expo', img_expo)
            cv2.imwrite('temp_img/8_img_expo.jpg', img_expo)
        return None
        
    
    def set_bound_and_blur(self, border_type = 'circle', border_center = (500, 500), radius = 500, border_rect = (8,8,73,73), border_color = [[18.9, 5.5], [17.7, 5.3], [24.6, 7.1]], ksize = (15,15), sig_x = 27,  show_img = False):
        border_center = (int(border_center[0]), int(border_center[1]))
        radius = int(radius)
        ksize = (g_utils.round_up_to_odd(ksize), g_utils.round_up_to_odd(ksize))
        sig_x = g_utils.round_up_to_odd(sig_x)
        
        boarder_map = np.zeros(self.syn_img_expo['tool_map'].shape, np.uint8)
        cv2.circle(boarder_map, center = border_center, radius=radius, color=1, thickness=-1)
        
        if border_type ==  'circle':
            img_boardered = np.zeros(self.syn_img_expo['img'].shape)
            for chn in range(0,3):
                img_boardered[:,:,chn] = self.syn_img_expo['img'][:,:,chn] * boarder_map + (1-boarder_map)*(border_color[chn][0] + border_color[chn][1]*np.random.randn(self.syn_img_expo['img'].shape[0], self.syn_img_expo['img'].shape[1]))     
            img_boardered = img_boardered.astype(np.uint8)
            tool_map_boardered = np.zeros(self.syn_img_expo['tool_map'].shape)
            tool_map_boardered = self.syn_img_expo['tool_map'] * boarder_map
        elif border_type == 'rect':
            img_boardered = np.zeros(self.syn_img_expo['img'].shape)
            tool_map_boardered = np.zeros(self.syn_img_expo['tool_map'].shape)
            boarder_map = np.zeros(self.syn_img_expo['tool_map'].shape)
            bd_top = int(border_rect[0])
            bd_bot = int(border_rect[1])
            bd_lft = int(border_rect[2])
            bd_rit = int(border_rect[3])
            
            boarder_map[bd_top:-bd_bot, bd_lft:-bd_rit] = 1
            img_boardered = g_utils.mask_mul(img = self.syn_img_expo['img'], mask = boarder_map) + g_utils.mask_mul(img = 15 * np.ones(self.syn_img_expo['img'].shape), mask = (1-boarder_map))
            tool_map_boardered = self.syn_img_expo['tool_map'] * boarder_map
            
        if sig_x > 0:
            sig_x = g_utils.round_up_to_odd(sig_x)
            img_blured = cv2.GaussianBlur(img_boardered, ksize, sig_x)
        else:
            img_blured = np.copy(img_boardered)
        
        self.syn_img_final = pd.Series({'index': self.syn_img_expo['index'],
                                        'img': img_blured,
                                        'type': self.syn_img_expo['type'],
                                        'contour': self.syn_img_expo['contour'],
                                        'tool_map': tool_map_boardered,
                                        'lines': self.syn_img_expo['lines'],
                                        'distance_map': self.syn_img_expo['distance_map'],
                                        'geo_feature_points': self.syn_img_expo['geo_feature_points'],
                                        'geo_feature_vectors': self.syn_img_expo['geo_feature_vectors'],
                                        'render_feature_points': self.syn_img_expo['render_feature_points'],
                                        'render_feature_vectors': self.syn_img_expo['render_feature_vectors'],
                                        'shrink_factors': self.syn_img_expo['shrink_factors'],
                                        'move_factors':self.syn_img_expo['move_factors'],
                                        'refl_factors': self.syn_img_expo['refl_factors'],
                                        'bkgd_idx': self.syn_img_expo['bkgd_idx'],
                                        'bkgd_img': self.syn_img_expo['bkgd_img'],
                                        'comb_factors': self.syn_img_expo['comb_factors'],
                                        'shadow_factors': self.syn_img_expo['shadow_factors'],
                                        'expo_factors': self.syn_img_expo['expo_factors'],
                                        'boarder_radius': radius,
                                        'boarder_map': boarder_map,
                                        'blur_factors':{'ksize': ksize,
                                                        'sig_x': sig_x}})
        
        self.syn_img_final_compressed = pd.Series({'index': self.syn_img_expo['index'],
                                        'type': self.syn_img_expo['type'],
                                        'lines': self.syn_img_expo['lines'],
                                        'geo_feature_points': self.syn_img_expo['geo_feature_points'],
                                        'geo_feature_vectors': self.syn_img_expo['geo_feature_vectors']})
        
        if show_img == True:
            g_utils.show_img('tool_map_boardered', tool_map_boardered*120)
            g_utils.show_img('img_blured', img_blured)
            cv2.imwrite('temp_img/9_img_blured.jpg', img_blured)
        if self.output_process == True:
            cv2.imwrite(self.folder_learn_iter + '/query_added/images/' + self.file_name + '_' + self.syn_type + '_' + '7_img_blured.jpg', img_blured)
        return None
    
    # [function] elastic_dilation_erosion
    # [Discription]: add elastic distortion, and dilation or erosion
    # [parameters]: elastic - if larger than 0, elastic distortion is applied
    # [parameters]: els_alpha - parameter of elastic distortion, recommend 2500
    # [parameters]: els_sigma - parameter of elastic distortion, recommend 15
    # [parameters]: dila_ero - if larger than 0, dilation is applied, if small than 0, erosion is applied, if equal to 0, none is applied
    # [parameters]: val_type - type, 0, 1 or 2, small diffrence 
    # [parameters]: dila_ero_size - int larger than 0, recommend 1-3
    # [result]: a pd.series containing the combined synthetic image and relative information including 
    # [return]: None
    def elastic_dilation_erosion(self, elastic, els_alpha, els_sigma, dila_ero, val_type, dila_ero_size, img_elastic, img_els_alpha, img_els_sigma):
        
        if (elastic > 0) and (self.random_tool_shape == 0):
            tool_map_new = np.zeros([self.syn_img_final['tool_map'].shape[0], self.syn_img_final['tool_map'].shape[1], 1])
            tool_map_new[:,:, 0] = self.syn_img_final['tool_map']
            tool_map_elas = g_utils.elastic_transform(tool_map_new, els_alpha, els_sigma, random_state=None)
            tool_map_new = tool_map_elas[:,:, 0]
        else:
            tool_map_new = self.syn_img_final['tool_map'][:]
        
        if dila_ero > 1:
            tool_map_new = g_utils.img_dilatation(tool_map_new, int(val_type), int(dila_ero_size))
        if dila_ero < -1:
            tool_map_new = g_utils.img_erosion(tool_map_new, int(val_type), int(dila_ero_size) + 1)
            
        if (img_elastic > 0) and (self.random_tool_shape == 0):
            img_elas = g_utils.elastic_transform(self.syn_img_final['img'], img_els_alpha, img_els_sigma, random_state=None)
        else:
            img_elas = self.syn_img_final['img'][:]
            
        tool_map_new = g_utils.img_dilatation(tool_map_new, 0, 1)
        tool_map_new = tool_map_new.clip(0,1)
        
        self.syn_img_final['img'] = img_elas[:]
        self.syn_img_final['tool_map'] = tool_map_new
        return None
    
    # old version of inpainting
    def img_inpaint2(self, img, label, tool_extract_map, bkgd = None, bkgd_percent = 0.5, fusion_type = 'avg', fusion_blur = 15, img_size = (240,240)):
        img = cv2.resize(img, img_size)
        label = cv2.resize(label, img_size)
        tool_extract_map = cv2.resize(tool_extract_map, img_size)
        
        color_adjust_strength = 0.9
        total_pixels = label.shape[0] * label.shape[1]
        non_zero_pixels = np.count_nonzero(label)
        
        if (non_zero_pixels/total_pixels) > 0.2 or g_utils.rand_from_range(0,1)<bkgd_percent:
            img_tool = bkgd.astype(np.float)
            img_bkgd = img.astype(np.float)
            # first, adjust the color style of the tool
            color_style_bkgd = np.array([np.mean(img_bkgd[:,:,0]), np.mean(img_bkgd[:,:,1]), np.mean(img_bkgd[:,:,2])])
            
            img_tool_chn0 = img_tool[:,:,0]
            img_tool_chn1 = img_tool[:,:,1]
            img_tool_chn2 = img_tool[:,:,2]
            color_style_tool = np.array([np.mean(img_tool_chn0[img_tool_chn0!=0]), np.mean(img_tool_chn1[img_tool_chn1!=0]), np.mean(img_tool_chn2[img_tool_chn2!=0])])
            
            color_adjust_factor = color_style_bkgd / color_style_tool
            
            
            img_tool_adjusted = np.zeros(img_tool.shape, np.float)
            for chn in range(0,3):
                img_tool_adjusted[:,:,chn] = img_tool[:,:,chn] * color_adjust_factor[chn]
                
            img_tool_adjusted = ((color_adjust_strength * img_tool_adjusted + (1 - color_adjust_strength) * img_tool)).astype(np.float).clip(0,255)
            
            
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
    
    def img_inpaint(self, img, label, tool_extract_map, bkgd , fusion_type = 'avg', fusion_blur = 15, img_size = (240,240)):
        img = cv2.resize(img, img_size)
        label = cv2.resize(label, img_size)
        tool_extract_map = cv2.resize(tool_extract_map, img_size)
        
        color_adjust_strength = 0.9

        img_tool = bkgd.astype(np.float)
        img_bkgd = img.astype(np.float)
        # first, adjust the color style of the tool
        color_style_bkgd = np.array([np.mean(img_bkgd[:,:,0]), np.mean(img_bkgd[:,:,1]), np.mean(img_bkgd[:,:,2])])
        
        img_tool_chn0 = img_tool[:,:,0]
        img_tool_chn1 = img_tool[:,:,1]
        img_tool_chn2 = img_tool[:,:,2]
        color_style_tool = np.array([np.mean(img_tool_chn0[img_tool_chn0!=0]), np.mean(img_tool_chn1[img_tool_chn1!=0]), np.mean(img_tool_chn2[img_tool_chn2!=0])])
        
        color_adjust_factor = color_style_bkgd / color_style_tool
        
        
        img_tool_adjusted = np.zeros(img_tool.shape, np.float)
        for chn in range(0,3):
            img_tool_adjusted[:,:,chn] = img_tool[:,:,chn] * color_adjust_factor[chn]
            
        img_tool_adjusted = ((color_adjust_strength * img_tool_adjusted + (1 - color_adjust_strength) * img_tool)).astype(np.float).clip(0,255)
        
        
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
        
        for chn in range(0,3):
            img_tool_adjusted[:,:,chn] = img_tool_adjusted[:,:,chn] * tool_extract_mask
            img_bkgd[:,:,chn] = img_bkgd[:,:,chn] * tool_map_reverse
            
        img_new = (img_bkgd + img_tool_adjusted).clip(0,255).astype(np.uint8)
            
        return img_new
    
    
    