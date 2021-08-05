# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 19:39:00 2020

@author: 75678
"""
import pandas as pd
import numpy as np
import cv2
import sys
import time
import glob
import random
import os

import tool_generator as gt
import generator_utils as g_utils

class data_frame_generator():
    
    folder_dataframe = None
    folder_background = None
    folder_tool = None
    
    df_tool = None # pandas data frame storing tools, see documentation for details
    df_background = None # pandas data frame storing backgrounds, see documentation for details
    
    idx_tool = None
    idx_background = None
    
    small_df = None

    # if start building data frame from zero, run this, 
    # if 'using_small_df = True', the images will be stored out side the data_frame file to save runnign memory
    def set_data_frame_folder(self, folder = 'tool_bkgd_data_frame', using_small_df = True):
        self.folder_dataframe = folder
        self.folder_background = folder + '/background'
        self.folder_tool = folder + '/tool'
        
        if using_small_df == True:
            self.small_df = True
        else:
            self.small_df = False
            
        try:
            os.mkdir(self.folder_dataframe)
        except:
            _= 1
        try:
            os.mkdir(self.folder_background)
        except:
            _= 1
        try:
            os.mkdir(self.folder_tool)
        except:
            _= 1
        
        return None

    # if start building data frame from zero, run this
    def init_tool_data_frame(self):
        if self.small_df == False:
            self.df_tool = pd.DataFrame(columns = ['index', 'img', 'type', 'contour', 'tool_map', 'lines', 'distance_map', 'geo_feature_points', 'geo_feature_vectors', 'render_feature_points', 'render_feature_vectors'])
        elif self.small_df == True:
            self.df_tool = pd.DataFrame(columns = ['index', 'type', 'contour', 'lines', 'distance_map', 'geo_feature_points', 'geo_feature_vectors', 'render_feature_points', 'render_feature_vectors'])
          
        self.idx_tool = 0 # if the data frame is initialized, set the index to be 0, if the data frame is loaded, the index will be changed by the loaded frame
        
        return None
    
    # if start building data frame from zero, run this
    def init_background_data_frame(self):
        if self.small_df == False:
            self.df_background = pd.DataFrame(columns = ['index', 'img', 'group', 'light_level'])
        elif self.small_df == True:
            self.df_background = pd.DataFrame(columns = ['index', 'group', 'light_level'])
        
        self.idx_background = 0
        
        return None
    
    # if start building data frame from existed csv file, run this and do not run init functions
    def load_tool_data_frame(self, file_path, file_type):
        if file_type == '.pkl':
            self.df_tool = pd.read_pickle(file_path, compression = 'xz')
        elif file_type == '.csv':
            self.df_tool = pd.read_csv(file_path)
        else:
            print('[DF Generator]: Unknow file type to load')
        self.idx_tool = self.df_tool.iloc[-1,0]
        self.idx_tool =  self.idx_tool + 1
        return None
    
    # if start building data frame from existed csv file, run this and do not run init functions
    def load_background_data_frame(self, file_path, file_type):
        if file_type == '.pkl':
            self.df_background = pd.read_pickle(file_path, compression = 'xz')
        elif file_type == '.csv':
            self.df_background = pd.read_csv(file_path)
        else:
            print('[DF Generator]: Unknow file type to load')
        self.idx_background = self.df_background.iloc[-1,0]
        self.idx_background = self.idx_background + 1
        return None
    
    
    # [function] tool_generate
    # [Discription]: using the tool generator to generate a fack tool with metal texture and contour & shape & geo points
    # [parameters]: texture_file_path - a texture back ground image, better to be square, featurePoints - a list of feature points; tool_type - associate with feature points 
    # [result]: add a new generated tool to the tool data frame
    # [return]: the tool generator and relative important factors
    def tool_generate(self, texture_file_path, feature_points, tool_type):
        tool_generator1 = gt.tool_generator()
    
        tool_generator1.load_img_tool_texture(texture_file_path)
        
        tool_generator1.contour_define(feature_points, tool_type)
        # tool_generator1.show_tool_contour()
        
        tool_generator1.contour_distance_map_gen()
        # tool_generator1.show_contour_distance_map()
        
        tool_generator1.tool_gen()
        
        # tool_img, tool_contour, tool_map, lines, contour_distance_map, geo_features = tool_generator1.output_result()
        
        if self.small_df == False:
            self.df_tool = self.df_tool.append({'index': self.idx_tool,
                                                'img': tool_generator1.tool_img,
                                                'type': tool_generator1.tool_type,
                                                'contour': tool_generator1.tool_contour,
                                                'tool_map': tool_generator1.tool_map,
                                                'lines': tool_generator1.lines,
                                                'distance_map': tool_generator1.contour_distance_map,
                                                'geo_feature_points': tool_generator1.geo_feature_points,
                                                'geo_feature_vectors': tool_generator1.geo_feature_vectors,
                                                'render_feature_points': tool_generator1.render_feature_points,
                                                'render_feature_vectors': tool_generator1.render_feature_vectors}, 
                                               ignore_index = True)
        elif self.small_df == True:
            self.df_tool = self.df_tool.append({'index': self.idx_tool,
                                                'type': tool_generator1.tool_type,
                                                'contour': tool_generator1.tool_contour,
                                                'lines': tool_generator1.lines,
                                                'distance_map': tool_generator1.contour_distance_map.astype(np.float16),
                                                'geo_feature_points': tool_generator1.geo_feature_points,
                                                'geo_feature_vectors': tool_generator1.geo_feature_vectors,
                                                'render_feature_points': tool_generator1.render_feature_points,
                                                'render_feature_vectors': tool_generator1.render_feature_vectors}, 
                                               ignore_index = True)
            cv2.imwrite(self.folder_tool + '/tool_' + str(self.idx_tool) + '.jpg',  tool_generator1.tool_img)
            cv2.imwrite(self.folder_tool + '/tool_map_' + str(self.idx_tool) + '.png',  tool_generator1.tool_map.astype(np.uint8))
        self.idx_tool = self.idx_tool + 1
        return  None
    
    # [function] tool_extract
    # [Discription]: cut out tools using labels, from data base randomly
    # [parameters]: data_base_folder - the path of the data base folder, should contain a folder images and a folder labels, and the files should have sam name. quantity - number of the exracted tools
    # [result]: add a new generated tool to the tool data frame
    # [return]: the tool generator and relative important factors
    def tool_extract(self, data_base_folder, quantity, extract_dilatation = 3):
        img_files = glob.glob(data_base_folder + '/' + '*.' + 'jpg')
        
        for i in range(0,quantity):
            img_path = random.choice(img_files)
            label_path = img_path[:-3] + 'png'

            tool_label = cv2.imread(label_path)
            tool_mask = tool_label[:]
            tool_mask_small = g_utils.img_dilatation(tool_mask, 0, int(extract_dilatation/2))
            # tool_mask_mid = g_utils.img_dilatation(tool_mask, 0, extract_dilatation + 2)
            tool_mask_large = g_utils.img_dilatation(tool_mask, 0, extract_dilatation)
            # tool_label = tool_label.clip(0, 1)
            tool_mask_small[tool_mask_small>0] = 1
            tool_mask_large[tool_mask_large>0] = 1
            tool_img = cv2.imread(img_path) * tool_mask_large
            
            if np.max(tool_img) <= 20:
                continue
                     
            # g_utils.show_img('img', tool_img)
            # g_utils.show_img('img', tool_label*200)
            
            self.df_tool = self.df_tool.append({'index': int(self.idx_tool),
                                                'type': -1,
                                                'contour': None,
                                                'lines': None,
                                                'distance_map': None,
                                                'geo_feature_points': None,
                                                'geo_feature_vectors': None,
                                                'render_feature_points': None,
                                                'render_feature_vectors': None}, 
                                               ignore_index = True)
            cv2.imwrite(self.folder_tool + '/tool_' + str(self.idx_tool) + '.bmp',  tool_img)
            cv2.imwrite(self.folder_tool + '/tool_map_' + str(self.idx_tool) + '.bmp',  tool_label)
            cv2.imwrite(self.folder_tool + '/tool_extract_mask_' + str(self.idx_tool) + '.bmp',  tool_mask_small)
            
            self.idx_tool = self.idx_tool + 1
    
    
    # [function] background_load
    # [Discription]: load background image and save it to dataframe
    # [parameters]: background_img_path - back ground image, better to be square, group - same group of backgrounds have similar style, which will be used for reflection rendering 
    # [result]: add a new background to the background data frame
    # [return]: None
    def background_load(self, background_img_path, group):
        
        bg_img = cv2.imread(background_img_path)
        if bg_img is None:
             print('[DF Generator] Error: Background image load failed, image does not exist')
             sys.exit()
             
        if np.abs((bg_img.shape[0]-bg_img.shape[1])/bg_img.shape[0]) > 0.2:
             print('[DF Generator] Error: Too large width & hight ratio. Background image should be close to square')
             sys.exit()
             
        bg_img = cv2.resize(bg_img, (1001,1001))
        
        light_level = np.zeros(3)
        for chn in range(0,3):
            light_level[chn] = np.average(bg_img[:,:,chn])
            
        self.df_background = self.df_background.append({'index': self.idx_background,
                                            'img': bg_img,
                                            'group': group,
                                            'light_level': light_level}, 
                                           ignore_index = True)
        
        self.idx_background = self.idx_background + 1    
        return None
    
    # [function] background_load_folder
    # [Discription]: load all background image from a folder and save it to dataframe, should notice that the images in the folder must be in one group
    # [parameters]: background_img_folder_path - back ground image folder (should contain '\'), all images are better to be square, group - same group of backgrounds have similar style, which will be used for reflection rendering 
    # [result]: add a new backgrounds to the background data frame
    # [return]: None
    def background_load_folder(self, background_img_folder_path, group, img_type = 'jpg'):
        if self.small_df == True:
            folder_group = self.folder_background + '/group_' + str(group)
            try:
                os.mkdir(folder_group)
            except:
                _= 1
        
        for img_path in glob.glob(background_img_folder_path + '*.' + img_type):
            bg_img = cv2.imread(img_path)
            if bg_img is None:
                 print('[DF Generator] Error: Background image load failed, image does not exist')
                 sys.exit()
                 
            if np.abs((bg_img.shape[0]-bg_img.shape[1])/bg_img.shape[0]) > 0.2:
                 print('[DF Generator] Error: Too large width & hight ratio. Background image should be close to square')
                 sys.exit()
                 
            # bg_img = cv2.resize(bg_img, (1001,1001))
            
            light_level = np.zeros(3)
            for chn in range(0,3):
                light_level[chn] = np.average(bg_img[:,:,chn])
                
            if self.small_df == False:
                self.df_background = self.df_background.append({'index': self.idx_background,
                                                    'img': bg_img,
                                                    'group': group,
                                                    'light_level': light_level}, 
                                                   ignore_index = True)
            elif self.small_df == True:
                self.df_background = self.df_background.append({'index': self.idx_background,
                                                    'group': group,
                                                    'light_level': light_level}, 
                                                   ignore_index = True)
                cv2.imwrite(folder_group + '/bkgd_' + str(self.idx_background) + '.jpg', bg_img)
            self.idx_background = self.idx_background + 1    
        return None
    
    # save tool data frame to a csv file, the folder path should include '\', 
    # if add_time_stamp = True, the saved csv file will contain a time stamp in the 
    # name and so that will not overwrite the old one
    def save_data_frame_tool(self, folder_path, add_time_stamp = False, file_type = '.pkl', file_type_param = None):
        try:
            os.mkdir(folder_path)
        except:
            _= 1
            
        if add_time_stamp == False:
            file_name = folder_path + '/data_frame_tool' + file_type
        else:
            file_name = folder_path + '/data_frame_tool_' + str(int(time.time())) + file_type
        
        if file_type == '.pkl':
            self.df_tool.to_pickle(file_name, compression = file_type_param)
        elif file_type == '.csv':
            self.df_tool.to_csv(file_name, index = False)
        elif file_type == '.h5':
            self.df_tool.to_hdf(file_name, key = 'a', mode = 'w')
        else:
            print('[DF Generator]: Unknown file type to save data')
        return None
    
    # save background data frame to a csv file, the folder path should include '\', 
    # if add_time_stamp = True, the saved csv file will contain a time stamp in the 
    # name and so that will not overwrite the old one
    def save_data_frame_background(self, folder_path, add_time_stamp = False, file_type = '.pkl', file_type_param = None):
        try:
            os.mkdir(folder_path)
        except:
            _= 1
            
        if add_time_stamp == False:
            file_name = folder_path + '/data_frame_background' + file_type
        else:
            file_name = folder_path + '/data_frame_background_' + str(int(time.time())) + file_type
        
        if file_type == '.pkl':
            self.df_background.to_pickle(file_name, compression = file_type_param)
        elif file_type == '.csv':
            self.df_background.to_csv(file_name, index = False)
        elif file_type == '.h5':
            self.df_background.to_hdf(file_name, key = 'a', mode = 'w')
        else:
            print('[DF Generator]: Unknown file type to save data')
        return None
    
    def show_df_background_size(self):
        print('[DF Generator]: Size of background dataframe:')
        print(self.df_background.shape)
        return None
    
    def show_df_tool_size(self):
        print('[DF Generator]: Size of tool dataframe:')
        print(self.df_tool.shape)
        return None
    
    