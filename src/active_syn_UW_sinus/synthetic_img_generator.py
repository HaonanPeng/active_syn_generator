# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 14:44:25 2020

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


import data_frame_generator as dfg
import generator_utils as g_utils

class syn_img_generator:
    
    index = None
    small_df = None
    
    df_tool = None # pandas data frame storing tools, see documentation for details
    df_background = None # pandas data frame storing backgrounds, see documentation for details
    
    pool_tool = None # pandas data frame storing selected tools, the tool will be randomly chosen from this pool
    pool_background = None # pandas data frame storing selected background, the tool will be randomly chosen from this pool
    
    tool = None # a pandas series, containing the image of the tool and relative imformation
    background = None # a pandas series, containing the image of the background and relative imformation
    
    tool_shrinked = None
    syn_img_final = None
    
    syn_img_final = None # final result of synthetic image
    syn_img_final_compressed = None
    
    folder_source = None
    folder_tool = None
    folder_background = None
    folder_dataframe = None
    folder_syn_imgs = None
    folder_ground_truth = None
    folder_img_n_gt = None
    time_stamp = None
    
    div_bkgd_files = None
    
    random_tool_shape = 0
    
    debug_temp = None
    
    # load both tool data frame and background data frame
    def load_src_dataframes(self, df_tool_path, df_background_path, small_df = True, source_path = 'tool_bkgd_data_frame', tool_types = None, background_groups = None, file_type = '.pkl', file_type_param = 'infer', diverse_bkgd_path = None):
        if file_type == '.pkl':
            df_tool = pd.read_pickle(df_tool_path, compression = file_type_param)
            df_background = pd.read_pickle(df_background_path, compression = file_type_param)
            
        self.small_df = small_df
        self.folder_source = source_path
        self.folder_tool = source_path + '/tool'
        self.folder_background = source_path + '/background'
        
        print('[SYIM Generator]: Tool data frame loaded, size:')
        print(df_tool.shape)
        print('[SYIM Generator]: Background data frame loaded, size:')
        print(df_background.shape)
        
        # set random pool-------------------------------
        if tool_types == None:
            self.pool_tool = df_tool
        else:
            self.pool_tool = pd.DataFrame(columns = df_tool.columns)
            for tool_type in tool_types:
                self.pool_tool = self.pool_tool.append(df_tool.loc[df_tool['type']==tool_type])
            
        if background_groups == None:
            self.pool_background = df_background
        else:
            self.pool_background = pd.DataFrame(columns = df_background.columns)
            for background_group in background_groups:
                self.pool_background = self.pool_background.append(df_background.loc[df_background['group']==background_group])
          
        # load all imgs and distance maps       
        # if self.small_df == True:
        #     self.pool_tool['img'] = ''
        #     self.pool_tool['tool_map'] = ''
        #     self.pool_tool['distance_map'] = ''
        #     self.pool_background['img'] = ''
            
        #     for tool in self.pool_tool:
        #         tool['img'] = cv2.imread(self.folder_tool + '/tool_' + str(self.tool['index']) + '.jpg')
        #         tool['tool_map'] = cv2.imread(self.folder_tool + '/tool_map_' + str(self.tool['index']) + '.png', cv2.IMREAD_GRAYSCALE)
        #         tool['distance_map'] = np.loadtxt(self.folder_tool + '/tool_distancemap_' + str(self.tool['index']) + '.txt')
        #         df_tool.loc[df_tool['index']==tool['index']] = tool
        if diverse_bkgd_path is not None:
            self.div_bkgd_files = glob.glob(diverse_bkgd_path + '/' + '*.' + 'jpg')
        print('[SYIM Generator]: Tool pool set, size:')
        print(self.pool_tool.shape)
        print('[SYIM Generator]: Background pool set, size:')
        print(self.pool_background.shape)
        
        return None
    
    def set_img_folder(self, folder_path = 'data_frame', add_time_stamp = False):
        if add_time_stamp == False :
            self.folder_dataframe = folder_path
            self.folder_syn_imgs = folder_path + '/syn_imgs'
            self.folder_ground_truth = folder_path + '/ground_truth'
            self.folder_ground_truth_rev = folder_path + '/ground_truth_rev'
            self.folder_ground_truth_rev2 = folder_path + '/ground_truth_rev2'
            self.folder_ground_truth_type1 = folder_path + '/ground_truth_type1'
            self.folder_ground_truth_type2 = folder_path + '/ground_truth_type2'
            self.folder_img_n_gt = folder_path + '/img_n_gt'
        else: 
            self.time_stamp = str(int(time.time()))
            self.folder_dataframe = folder_path + '_' + self.time_stamp
            self.folder_syn_imgs = self.folder_dataframe + '/syn_imgs' 
            self.folder_ground_truth = self.folder_dataframe + '/ground_truth'
            self.folder_ground_truth_rev = self.folder_dataframe + '/ground_truth_rev'
            self.folder_ground_truth_rev2 = self.folder_dataframe + '/ground_truth_rev2'
            self.folder_ground_truth_type1 = self.folder_dataframe + '/ground_truth_type1'
            self.folder_ground_truth_type2 = self.folder_dataframe + '/ground_truth_type2'
            self.folder_img_n_gt = self.folder_dataframe + '/img_n_gt' 
        
        try:
            os.mkdir(self.folder_dataframe)
        except:
            _= 1
        try:
            os.mkdir(self.folder_syn_imgs)
        except:
            _= 1
        try:
            os.mkdir(self.folder_ground_truth)
        except:
            _= 1
        try:
            os.mkdir(self.folder_ground_truth_rev)
        except:
            _= 1
        try:
            os.mkdir(self.folder_ground_truth_rev2)
        except:
            _= 1
        try:
            os.mkdir(self.folder_ground_truth_type1)
        except:
            _= 1
        try:
            os.mkdir(self.folder_ground_truth_type2)
        except:
            _= 1
        try:
            os.mkdir(self.folder_img_n_gt)
        except:
            _= 1
               
        return None
    
    def init_syn_img_dataframe(self, start_index = 0):
        self.index = start_index
        
        self.df_syn_img = pd.DataFrame(columns = ['index',
                                                    'img',
                                                    'type',
                                                    'contour',
                                                    'tool_map',
                                                    'lines',
                                                    'distance_map',
                                                    'geo_feature_points',
                                                    'geo_feature_vectors',
                                                    'render_feature_points',
                                                    'render_feature_vectors',
                                                    'shrink_factors',
                                                    'move_factors',
                                                    'refl_factors',
                                                    'bkgd_idx',
                                                    'bkgd_img',
                                                    'comb_factors',
                                                    'shadow_factors',
                                                    'expo_factors',
                                                    'boarder_radius',
                                                    'blur_factors'])
        
        self.df_syn_img_compressed = pd.DataFrame(columns = ['index',
                                                            'type',
                                                            'lines',
                                                            'geo_feature_points',
                                                            'geo_feature_vectors'])
        return None
    
    # Not longer used! Combined in load_src_dataframes
    # [function] set_random_pool
    # [Discription]: set the random pool of the tools and backgrounds
    # [parameters]: tool_types - a list of intergers of tool types, if none, use all types, background_groups - a list of intergers of background groups, if none, use all groups
    # [result]: set up random pools of tools and backgrounds
    # [return]: None
    def set_random_pool(self, tool_types = None, background_groups = None):
        
        if tool_types == None:
            self.pool_tool = self.df_tool
        else:
            self.pool_tool = pd.DataFrame(columns = self.df_tool.columns)
            for tool_type in tool_types:
                self.pool_tool = self.pool_tool.append(self.df_tool.loc[self.df_tool['type']==tool_type])
            
        if background_groups == None:
            self.pool_background = self.df_background
        else:
            self.pool_background = pd.DataFrame(columns = self.df_background.columns)
            for background_group in background_groups:
                self.pool_background = self.pool_background.append(self.df_background.loc[self.df_background['group']==background_group])
        
        
        
        print('[SYIM Generator]: Tool pool set, size:')
        print(self.pool_tool.shape)
        print('[SYIM Generator]: Background pool set, size:')
        print(self.pool_background.shape)
        
        return None
    
    # select a tool from pool, if none index given, it will be randomly chosen
    def select_tool(self, index = None, flip = 0.5, alpha = 991, sigma = 9, use_random_shape = -0.5, control_points = 5, rad = 0.2, edgy = 0.05, show_img = False):
        control_points = int(control_points)
        
        if use_random_shape < 0:
            if index == None:
                self.tool = self.pool_tool.sample().squeeze()
            else:
                self.tool = self.pool_tool.iloc[index].squeeze()
                
            if self.small_df == True:
                self.tool['img'] = cv2.imread(self.folder_tool + '/tool_' + str(int(self.tool['index'])) + '.bmp')
                # self.tool['img'] = cv2.imread('D:/uw/PHD/2020summer/instrument segmentation/code_dev/synthetic_endo_img_generator_v3/tool_bkgd_data_frame_3/tool/tool_1.jpg')
                tool_map = cv2.imread(self.folder_tool + '/tool_map_' + str(int(self.tool['index'])) + '.bmp', cv2.IMREAD_GRAYSCALE)
                tool_extract_mask = cv2.imread(self.folder_tool + '/tool_extract_mask_' + str(int(self.tool['index'])) + '.bmp', cv2.IMREAD_GRAYSCALE)
                self.tool['tool_map'] = np.zeros((tool_map.shape[0], tool_map.shape[1],2))
                self.tool['tool_map'][:,:,0] = tool_map
                self.tool['tool_map'][:,:,1] = tool_extract_mask
                # self.tool['distance_map'] = np.loadtxt(self.folder_tool + '/tool_distancemap_' + str(self.tool['index']) + '.txt')
              
            if flip < 0:
                self.tool['img'] = cv2.flip(self.tool['img'], 1)
                self.tool['tool_map'] = cv2.flip(self.tool['tool_map'], 1)
                # self.tool['distance_map'] = cv2.flip(self.tool['distance_map'].astype(np.int16), 0)
            
            # distort image
            # tool_n_map = np.zeros([self.tool['tool_map'].shape[0], self.tool['tool_map'].shape[1], 5])
            # tool_n_map[:,:, 0:3] = self.tool['img']
            # tool_n_map[:,:, 3] = self.tool['tool_map']
            # tool_n_map[:,:, 4] = self.tool['distance_map'][1000:2001, 1000:2001]
            # tool_n_map = g_utils.elastic_transform(tool_n_map, alpha, sigma, random_state=None)
            # self.tool['img'] = tool_n_map[:,:, 0:3]
            # self.tool['tool_map'] = tool_n_map[:,:, 3]
            # self.tool['distance_map'][1000:2001, 1000:2001] = tool_n_map[:,:, 4]
        else:
            self.random_tool_shape = 1
            self.tool = self.pool_tool.sample().squeeze()
            if self.small_df == True:
                self.tool['img'] = cv2.imread(self.folder_tool + '/tool_' + str(self.tool['index']) + '.jpg')
                self.tool['tool_map'] = cv2.imread(self.folder_tool + '/tool_map_' + str(self.tool['index']) + '.png', cv2.IMREAD_GRAYSCALE)
            
            tool_map = g_utils.random_shape(control_points = 5, rad = 0.2, edgy = 0.05 )
    
            tool_img = np.ones(self.tool['img'].shape)  * np.array([255 * np.random.rand(), 255 * np.random.rand(), 255 * np.random.rand()])
            tool_img = (tool_img + (np.random.rand(1001,1001,3)-0.5)*150)
            for chn in range(0,3):
                tool_img[:,:,chn] =  tool_img[:,:,chn]*tool_map
            tool_img = tool_img.clip(0,255).astype(np.uint8)
            

            
        # tool_img = self.tool['img']
        # tool_img[tool_img==0] = 200
        if show_img == True:
            tool_img = self.tool['img']
            tool_img[tool_img==0] = 255
            g_utils.show_img('selected_tool', tool_img)
            cv2.imwrite('temp_img/1_selected_tool.jpg', self.tool['img'])
            
    # select a background from pool, if none index given, it will be randomly chosen        
    def select_background(self, index = None, use_diverse_bkgd = -0.5, show_img = False):
        if index == None:
            self.background = self.pool_background.sample().squeeze()
        else:
            self.background = self.pool_background.iloc[index].squeeze()
            
        if self.small_df == True:
            self.background['img'] = cv2.imread(self.folder_background + '/group_' + str(self.background['group']) + '/bkgd_' + str(self.background['index']) + '.jpg')
            
        if use_diverse_bkgd > 0:
            div_bkgd_path = random.choice(self.div_bkgd_files)
            self.background['img'] = cv2.resize(cv2.imread(div_bkgd_path), (self.tool['img'].shape[0], self.tool['img'].shape[1]))
        self.background['img'] = g_utils.rotate_image(self.background['img'], g_utils.rand_from_range(0, 359))
            
        if show_img == True:
            g_utils.show_img('selected_bkgd', self.background['img'])
            cv2.imwrite('temp_img/2_selected_bkgd.jpg', self.background['img'])
    
    # [function] tool_shrink
    # [Discription]: shrink the tool's size, as well as the relative information of points and vectors. The shinked tool will still in the center of the image, instead of shrinking to upper left side
    # [parameters]: factor - the shrinking factor, must be in (0,1)
    # [result]: a pd.series containing shringked image and relative information including shrinked feature points
    # [return]: None
    def shrink_tool(self, factor, show_img = False):
        # tool_shape_org = self.tool['img'].shape
        
        # shape_sk = g_utils.round_up_to_odd(factor * tool_shape_org[0])
        # factor = shape_sk/tool_shape_org[0]

        # tool_shrinked = cv2.resize(self.tool['img'], (shape_sk, shape_sk))
        # img_new = np.zeros(self.tool['img'].shape, np.uint8)
        # img_new[int(0.5*(tool_shape_org[0] - shape_sk)) : int(0.5*(tool_shape_org[1] + shape_sk)), int(0.5*(tool_shape_org[1] - shape_sk)) : int(0.5*(tool_shape_org[0] + shape_sk)) , :] = tool_shrinked
        
        # tool_map_int = self.tool['tool_map'].astype(np.uint8)
        # tool_map_shrinked = cv2.resize(tool_map_int, (shape_sk, shape_sk))
        # tool_map_new = np.zeros(self.tool['tool_map'].shape, np.uint8)
        # tool_map_new[int(0.5*(tool_shape_org[0] - shape_sk)) : int(0.5*(tool_shape_org[1] + shape_sk)), int(0.5*(tool_shape_org[1] - shape_sk)) : int(0.5*(tool_shape_org[0] + shape_sk))] = tool_map_shrinked
        
        # movement = int(0.5*(tool_shape_org[0] - shape_sk))
        
        tool_shape_org = self.tool['img'].shape
        
        shape_sk = g_utils.round_up_to_odd(factor * tool_shape_org[0])
        factor = shape_sk/tool_shape_org[0]

        tool_shrinked = cv2.resize(self.tool['img'], (shape_sk, shape_sk))
        img_new = np.zeros(self.tool['img'].shape, np.uint8)
        img_new[int(0.5*(tool_shape_org[0] - shape_sk)) : int(0.5*(tool_shape_org[1] + shape_sk)), int(0.5*(tool_shape_org[1] - shape_sk)) : int(0.5*(tool_shape_org[0] + shape_sk)) , :] = tool_shrinked
        
        tool_map_int = self.tool['tool_map'].astype(np.uint8)
        tool_map_shrinked = cv2.resize(tool_map_int, (shape_sk, shape_sk))
        tool_map_new = np.zeros(self.tool['tool_map'].shape, np.uint8)
        tool_map_new[int(0.5*(tool_shape_org[0] - shape_sk)) : int(0.5*(tool_shape_org[1] + shape_sk)), int(0.5*(tool_shape_org[1] - shape_sk)) : int(0.5*(tool_shape_org[0] + shape_sk))] = tool_map_shrinked
        
        movement = int(0.5*(tool_shape_org[0] - shape_sk))
        
       
        
        
        self.tool_shrinked = pd.Series({'index': self.index,
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
        
        # g_utils.show_tool_map('12',self.tool_shrinked['tool_map'])
        # g_utils.show_contour_distance_map('test_dismap', self.tool['distance_map'], self.tool['contour'])
        # g_utils.show_contour_distance_map('test_dismap', self.tool_shrinked['distance_map'], self.tool_shrinked['contour'])
        # img_temp = g_utils.draw_vectors(img_new, [self.tool_shrinked['geo_feature_points'][1], self.tool_shrinked['geo_feature_points'][3], self.tool_shrinked['geo_feature_points'][2]], self.tool_shrinked['geo_feature_vectors'], color = (100,200,0))
        # g_utils.show_img('12',img_temp)
        # temp_img = g_utils.draw_points(img_new , np.int_(np.array(self.tool['geo_feature_points'])*factor + movement))
        # g_utils.show_img('test', temp_img)
        # self.debug_temp = self.tool_shrinked
        # g_utils.show_img('11',tool_shrinked)
        # g_utils.show_img('12',img_new)
        
        return None
    
    
    # [function] move_tool
    # [Discription]: move the tool by factor*shape_shrinked, notice that the tool will be first moved to touch the right side of the image so that the end edge of the tool will not be seen in the image, and then the tool is rotated, along with all the relative maps 
    # [parameters]: x_factor, y_factor - the moving factor. 'y_factor' must be non-negative If y_factor is larger than 1, the tool image will be moved out of the final image. 0 means that the tool image will not move in that direction, and 1 means that the tool image will move out of the final snythetic image
    # [parameters]: r_factor - the rotation factor in degree, should be in [0,360)
    # [result]: a pd.series containing moved image and relative information including moved feature points
    # [return]: None
    def move_tool(self, x_factor, y_factor, r_factor, show_img = False):
        # first, if the tool is moved out of the image, set the image to be zeros
        img_shape = self.tool_shrinked['img'].shape
        shrinked_shape = self.tool_shrinked['shrink_factors']['shape_shrinked']
        x_movement = int( x_factor * shrinked_shape )
        y_movement = int( y_factor * shrinked_shape )
        
        T_mat = np.array([[1.0, 0.0, y_movement + 0.5*(img_shape[0] - shrinked_shape)],
                          [0.0, 1.0, x_movement]])
        
        R_mat = cv2.getRotationMatrix2D((0.5*img_shape[0], 0.5*img_shape[1]), r_factor, scale=1) # these T matrix and R matrix are for opencv, not for translating and rotating points and vectors, they are defined later
        
        if  y_movement >= (0.5*( img_shape[0] + shrinked_shape)) or abs(x_movement) >= img_shape[0]:
            img_new = np.zeros(self.tool_shrinked['img'].shape, np.uint8)
            tool_map_new = np.zeros(self.tool_shrinked['tool_map'].shape, np.uint8)
        else:
            # translate and rotate the image
            img_new = cv2.warpAffine(self.tool_shrinked['img'], T_mat, (img_shape[0], img_shape[1])) 
            img_new = cv2.warpAffine(img_new, R_mat, (img_shape[0], img_shape[1])) 
            # translate and rotate the tool_map
            tool_map_new = cv2.warpAffine(self.tool_shrinked['tool_map'], T_mat, (img_shape[0], img_shape[1])) 
            tool_map_new = cv2.warpAffine(tool_map_new, R_mat, (img_shape[0], img_shape[1])) 
        
        # translate and rotate the distance map
        # R_mat_dismap = cv2.getRotationMatrix2D((1500,1500), r_factor, scale=1)
        # distance_map_new = cv2.warpAffine(self.tool_shrinked['distance_map'], T_mat, (3001, 3001)) 
        # distance_map_new = cv2.warpAffine(distance_map_new, R_mat_dismap, (3001, 3001)) 
         
        # translate and rotate geo_feature_points
        # geo_feature_points = self.tool_shrinked['geo_feature_points'] + [x_movement, y_movement + 0.5*(1001-shrinked_shape)]
        
        # geo_feature_points = np.append(geo_feature_points.T, np.ones((1,self.tool_shrinked['geo_feature_points'].shape[0])), axis=0)
        
        # r_factor_rad = r_factor/180 * np.pi
        # R_mat_p = np.array([[np.cos(r_factor_rad), -np.sin(r_factor_rad), -np.cos(r_factor_rad)*500 + np.sin(r_factor_rad)*500 + 500],
        #                     [np.sin(r_factor_rad),  np.cos(r_factor_rad), -np.sin(r_factor_rad)*500 - np.cos(r_factor_rad)*500 +500]])

        # geo_feature_points_new = np.int_(np.dot(R_mat_p, geo_feature_points)).T
        
        # # translate and rotate render_feature points
        # render_feature_points = self.tool_shrinked['render_feature_points'] + [x_movement, y_movement + 0.5*(1001-shrinked_shape)]
        # render_feature_points = np.append(render_feature_points.T, np.ones((1,self.tool_shrinked['render_feature_points'].shape[0])), axis=0)
        # render_feature_points_new = np.int_(np.dot(R_mat_p, render_feature_points)).T
        
        # translate and rotate line points
        # lines_new = []
        # for line in self.tool['lines']:
        #     line_new = line + [x_movement, y_movement + 0.5*(1001-shrinked_shape)]
        #     line_new = np.append(line_new.T, np.ones((1,line_new.shape[0])), axis=0)
        #     line_new = np.int_(np.dot(R_mat_p, line_new)).T
        #     lines_new.append(line_new)
        
        # # translate geo_feature vectors
        # R_mat_v = np.array([[np.cos(r_factor_rad), -np.sin(r_factor_rad)], # 2x2 rotation matrix for rotating vectors
        #                     [np.sin(r_factor_rad),  np.cos(r_factor_rad)]])
        # geo_feature_vectors_new = (np.dot(R_mat_v , np.array(self.tool_shrinked['geo_feature_vectors']).T)).T
        # # translate render_feature vectors
        # render_feature_vectors_new = (np.dot(R_mat_v , np.array(self.tool_shrinked['render_feature_vectors']).T)).T
        
        # # translate and rotate the tool contour
        # contour_new = self.tool_shrinked['contour'] + [x_movement, y_movement + 0.5*(1001-shrinked_shape)]
        # contour_new = np.append(contour_new.T, np.ones((1,self.tool_shrinked['contour'].shape[0])), axis=0)
        # contour_new = np.int_(np.dot(R_mat_p, contour_new)).T
        
        
        
        
        
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
                                        'move_factors':{'x_movement': x_movement, 'y_movement': y_movement + 0.5*(1001-shrinked_shape), 'rotate_degree': r_factor}})

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
        return None
    
    
    # [function] render_reflection
    # [Discription]: render the reflection of the tool, a background from the same group of backgrounds will be chosen and render the metal texture of the tool
    # [parameters]: refl_type - if 0, there will be no reflection rendered. refl_param - a list of numbers wokring with reflection type, if None is given, defult parameters will be used
    # [result]: a pd.series contains rendered image and the index of the relfection background
    # [return]: None
    def render_reflection(self, index = None, refl_type = 1, refl_param = None, use_diverse_refl = -0.5, diverse_bkgd_path = 'val2017', show_img = False):
        # if use_diverse_refl < 0:
        #     if index == None:
        #         bkgd_refl = self.pool_background.loc[self.pool_background['group']==self.background['group']].sample().squeeze()
        #     else:
        #         bkgd_refl = self.pool_background.loc[self.pool_background['group']==self.background['group']].iloc[index]
        #     if self.small_df == True:
        #         bkgd_refl['img'] = cv2.imread(self.folder_background + '/group_' + str(bkgd_refl['group']) + '/bkgd_' + str(bkgd_refl['index']) + '.jpg')
        # else:
        #     if index == None:
        #         bkgd_refl = self.pool_background.loc[self.pool_background['group']==self.background['group']].sample().squeeze()
        #     else:
        #         bkgd_refl = self.pool_background.loc[self.pool_background['group']==self.background['group']].iloc[index]
        #     if self.small_df == True:
        #         bkgd_refl['img'] = cv2.imread(self.folder_background + '/group_' + str(bkgd_refl['group']) + '/bkgd_' + str(bkgd_refl['index']) + '.jpg')
            
        #     div_bkgd_files = glob.glob(diverse_bkgd_path + '/' + '*.' + 'jpg')
        #     div_bkgd_path = random.choice(div_bkgd_files)
        #     bkgd_refl['img'] = cv2.resize(cv2.imread(div_bkgd_path), (1001,1001))
        
        
        # if refl_type == 0 :
        #     img_new = self.tool_moved['img']
        # elif refl_type == 1 :
        #     if refl_param == None:
        #         tool_factor = 0.8
        #         bkgd_factor = 0.3
        #     else:
        #         tool_factor = refl_param[0]
        #         bkgd_factor = refl_param[1]            
            
        #     img_bkgd_refl = bkgd_refl['img']
        #     img_bkgd_refl = cv2.rotate(img_bkgd_refl, cv2.ROTATE_90_CLOCKWISE)
        #     tool_map = self.tool_moved['tool_map']
            
        #     img_refl = np.zeros(self.tool_moved['img'].shape)
        #     for chn in range (0,3):
        #         img_refl[:,:,chn] = img_bkgd_refl[:,:,chn] * tool_map
            
        #     # g_utils.show_img('tool', self.tool_moved['img'])
        #     # g_utils.show_img('img_refl', img_refl)
        #     img_new = (tool_factor * self.tool_moved['img'] + bkgd_factor * img_refl).clip(0,255).astype(np.uint8)
        # elif refl_type == 2:
        #     if refl_param == None:
        #         tool_factor = 1
        #         bkgd_factor = 1
        #     else:
        #         tool_factor = refl_param[0]
        #         bkgd_factor = refl_param[1]
                
        #     img_new = np.zeros(self.tool_moved['img'].shape)
        #     img_bkgd = bkgd_refl['img']
        #     img_bkgd = cv2.rotate(img_bkgd, cv2.ROTATE_90_CLOCKWISE)
        #     normed_bkgd = 2 * bkgd_factor * img_bkgd/(np.max(img_bkgd) - np.min(img_bkgd)) + (1 - bkgd_factor)
        #     for chn in range (0,3):
        #         img_new[:,:,chn] = tool_factor * self.tool_moved['img'][:,:,chn] * normed_bkgd[:,:,chn]
        #     img_new = img_new.clip(0,255).astype(np.uint8)
        # else:
        #     print('[SYIM Generator]Error: Unknown reflection rendering type.')
        
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
        
        return None
    
    # [function] combine_syn_img
    # [Discription]: combine the tool image and the background, adjusting the color style of the tool to match the color style of the background
    # [parameters]: color_adjust_factor - if 1 the tool's color will be adjust to the same BGR ratio as the background, if 0, the tool's color will not be adjusted
    # [result]: a pd.series containing the combined synthetic image and relative information including 
    # [return]: None
    def combine_syn_img(self, color_adjust_strength = 0.8, brightness_adjust = 1, border_fusion = False, fusion_blur = (50,50), show_img = False): 
        
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
            img_tool_adjusted[img_tool_adjusted == 0] = 200
            
            tool_mask_blur = tool_extract_mask.astype(np.float32)  #+ tool_map
            tool_mask_blur = tool_mask_blur * 100
            # tool_mask_blur = cv2.GaussianBlur(tool_mask_blur, (15,15), 0)
            tool_mask_blur = cv2.blur(tool_mask_blur, (50,50))
            tool_extract_mask = tool_mask_blur/100
        
        tool_map_reverse = np.ones(tool_extract_mask.shape, np.float32) - tool_extract_mask
        
        # contour_map = -tool_map_reverse - self.tool_refl['tool_map']
        # contour_map[contour_map == 0] = 1
        # contour_map[contour_map < 0] = 0
        # tool_map_reverse = tool_map_reverse + contour_map
        
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
        return None
    
    
    # [function] render_shadow
    # [Discription]: render shadow on both tool and background
    # [parameters]: range - pixels that have distance larger than the range will not have rendered shadow, 
    # [result]: a pd.series containing the combined synthetic image and relative information including 
    # [return]: None
    def render_shadow(self, rander_shadow = 0.5, tool_shadow_range = 100, tool_shadow_brightness = 0.5, bkgd_shadow_range = 20, bkgd_shadow_brightness = 0.1, show_img = False):
        # dis_map = self.syn_img_comb['distance_map'][1000:2001,1000:2001]
        # self.debug_temp = dis_map
        # g_utils.show_contour_distance_map('distance_map', self.syn_img_comb['distance_map'], self.syn_img_comb['contour'])
        if (rander_shadow == 0.5) and (self.random_tool_shape < 0):
            dis_map_tool_shad = np.copy(self.syn_img_comb['distance_map'][1000:2001,1000:2001] + 2.0) # '+2' here is to prevent the contour of the tool is not included in the shadow map
            
            dis_map_tool_shad[dis_map_tool_shad <= 0] = -100000
            
            dis_map_bkgd_shad = -np.copy(self.syn_img_comb['distance_map'][1000:2001,1000:2001])
            dis_map_bkgd_shad[dis_map_bkgd_shad > bkgd_shadow_range] = 0
            dis_map_bkgd_shad[dis_map_bkgd_shad <= 0] = -100000
            
            ## This is the reflection method 1
            # tool_shad_mat = (1 * tool_shadow_brightness + dis_map_tool_shad  / (np.max(dis_map_tool_shad) + tool_shadow_range)).clip(0,1)
            # tool_shad_mat[tool_shad_mat==0] = 1
            # tool_shad_mat = np.sqrt(tool_shad_mat)
            
            ## This is the reflection methond 2
            tool_shad_mat = (0.5 * np.pi * (1 * tool_shadow_brightness + dis_map_tool_shad  / (np.max(dis_map_tool_shad) + tool_shadow_range))).clip(0, 0.5* np.pi)
            tool_shad_mat = np.sin(tool_shad_mat)
            tool_shad_mat[tool_shad_mat==0] = 1
            tool_shad_mat = np.sqrt(tool_shad_mat)
            
            bkgd_shad_mat = (bkgd_shadow_brightness + dis_map_bkgd_shad  / bkgd_shadow_range).clip(0,1)
            bkgd_shad_mat[bkgd_shad_mat==0] = 1
            
            if bkgd_shadow_range <= 0:
                bkgd_shad_mat = np.ones(bkgd_shad_mat.shape)
            
            img_new = np.zeros(self.syn_img_comb['img'].shape)
    
            for chn in range(0,3):
                img_new[:,:,chn] = (self.syn_img_comb['img'][:,:,chn] * tool_shad_mat * bkgd_shad_mat).clip(0,255)
            img_new = img_new.astype(np.uint8)
            
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
        else:
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
        if (render_exposure == 0.5) and (self.random_tool_shape < 0):
            expo_spread = int(expo_spread)
            
            render_vectors = g_utils.normalize_vectors(self.syn_img_shad['render_feature_vectors'])
            vector_mid = np.array(render_vectors[2])
            vector_mid_vert = np.array([vector_mid[1], -vector_mid[0]])
            start_point = self.syn_img_shad['render_feature_points'][2] + expo_move[0]*vector_mid + expo_move[1]*vector_mid_vert
            
            r_factor_rad = expo_angle/360 * np.pi
            R_mat_v = np.array([[np.cos(r_factor_rad), -np.sin(r_factor_rad)], # 2x2 rotation matrix for rotating vectors
                                [np.sin(r_factor_rad),  np.cos(r_factor_rad)]])
            expo_vector = (np.dot(R_mat_v , np.array(self.syn_img_shad['geo_feature_vectors'][2]).T)).T
            
            [expo_vector_norm] = g_utils.normalize_vectors([expo_vector])
            expo_vector_norm = np.array(expo_vector_norm)
            vert_vector = np.array([expo_vector_norm[1], - expo_vector_norm[0]])
            
            expo_map = np.ones(self.syn_img_shad['tool_map'].shape)
            width_cur_l = 0.5 * expo_width
            width_cur_r = 0.5 * expo_width
            for distance in range(0, expo_lenth):
                center_point = (start_point + distance * expo_vector_norm).clip(0,1000)
                
                width_cur_l = width_cur_l + np.random.randn()*0.02*expo_width
                width_cur_r = width_cur_r + np.random.randn()*0.02*expo_width
                
                taper_factor = expo_taper[0] + distance/expo_lenth * (expo_taper[1] - expo_taper[0])
                
                expo_line_p1 = np.int_(center_point + taper_factor * width_cur_l * vert_vector)
                expo_line_p2 = np.int_(center_point - taper_factor * width_cur_r * vert_vector)
                
                cv2.line(expo_map, (expo_line_p1[1], expo_line_p1[0]), (expo_line_p2[1], expo_line_p2[0]), color = expo_strength)
            
            expo_map = cv2.GaussianBlur(expo_map,(51,51),expo_spread)
            expo_map = cv2.GaussianBlur(expo_map,(51,51),expo_spread)
            
            # expo_map_show = (expo_map * 255 / expo_strength).clip(0,255).astype(np.uint8)
            # expo_map_show = cv2.GaussianBlur(expo_map_show, (101,101), 51)
            
            
            img_expo = np.float_(self.syn_img_shad['img'])
            for chn in range(0,3):
                img_expo[:,:,chn] = img_expo[:,:,chn] * expo_map
            img_expo = img_expo.clip(0,255).astype(np.uint8)
            
            
            
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
        else:
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
        
    
    def set_bound_and_blur(self, border_center = (500, 500), radius = 500, border_color = [[18.9, 5.5], [17.7, 5.3], [24.6, 7.1]], ksize = (15,15), sig_x = 27,  show_img = False):
        border_center = (int(border_center[0]), int(border_center[1]))
        radius = int(radius)
        ksize = (g_utils.round_up_to_odd(ksize[0]), g_utils.round_up_to_odd(ksize[1]))
        sig_x = g_utils.round_up_to_odd(sig_x)
        
        boarder_map = np.zeros(self.syn_img_expo['tool_map'].shape, np.uint8)
        cv2.circle(boarder_map, center = border_center, radius=radius, color=1, thickness=-1)
        
        img_boardered = np.zeros(self.syn_img_expo['img'].shape)
        for chn in range(0,3):
            img_boardered[:,:,chn] = self.syn_img_expo['img'][:,:,chn] * boarder_map + (1-boarder_map)*(border_color[chn][0] + border_color[chn][1]*np.random.randn(self.syn_img_expo['img'].shape[0], self.syn_img_expo['img'].shape[1]))
            
        img_boardered = img_boardered.astype(np.uint8)
        img_blured = cv2.GaussianBlur(img_boardered, ksize, sig_x)
        
        tool_map_boardered = np.zeros(self.syn_img_expo['tool_map'].shape)
        tool_map_boardered = self.syn_img_expo['tool_map'] * boarder_map
        
        
        
        
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
    
    
    # add the latest rendered syn_img to and relative information to dataframe, and save the image and ground truth masks to the folders    
    def add_to_dataframe(self, img_size = (1001,1001)):

        self.df_syn_img_compressed = self.df_syn_img_compressed.append(self.syn_img_final_compressed, ignore_index=True)
        
        cv2.imwrite(self.folder_syn_imgs + '/syn_img_' + str(self.syn_img_final['index']) + '.jpg', cv2.resize(self.syn_img_final['img'], img_size))
        
        tool_map_boardered = self.syn_img_final['tool_map'][:]
        boarder_map = self.syn_img_final['boarder_map']
        
        tool_map_rev = np.ones(self.syn_img_final['tool_map'].shape) - tool_map_boardered
        tool_map_rev2 = tool_map_rev * boarder_map
        
        # 3 classes type 1,tool is 0, background is 2, black boarder is 1
        rev_tool_map_boardered = np.zeros(tool_map_boardered.shape)
        rev_tool_map_boardered = 2*(1 - tool_map_boardered)          
        rev_boarder_map = 1 - boarder_map
        rev_tool_map_boardered_type1 = (rev_tool_map_boardered - rev_boarder_map).clip(0,2).astype(np.uint8)
 
        # # 3 classes type 2, tool is 2, background is 1 and black boarder is 0
        rev_tool_map_boardered = tool_map_boardered + 1
        rev_tool_map_boardered = rev_tool_map_boardered * boarder_map
        rev_tool_map_boardered_type2 = rev_tool_map_boardered
        rev_tool_map_boardered_type2 = rev_tool_map_boardered_type2.clip(0,2).astype(np.uint8)
        rev_tool_map_boardered_type2[rev_tool_map_boardered_type2>2] = 1
        
        cv2.imwrite(self.folder_ground_truth + '/syn_img_' + str(self.syn_img_final['index']) + '.png', cv2.resize(self.syn_img_final['tool_map'], img_size))
        cv2.imwrite(self.folder_ground_truth_rev + '/syn_img_' + str(self.syn_img_final['index']) + '.png', cv2.resize(tool_map_rev, img_size))
        cv2.imwrite(self.folder_ground_truth_rev2 + '/syn_img_' + str(self.syn_img_final['index']) + '.png', cv2.resize(tool_map_rev2, img_size))
        cv2.imwrite(self.folder_ground_truth_type1 + '/syn_img_' + str(self.syn_img_final['index']) + '.png', cv2.resize(rev_tool_map_boardered_type1, img_size))
        cv2.imwrite(self.folder_ground_truth_type2 + '/syn_img_' + str(self.syn_img_final['index']) + '.png', cv2.resize(rev_tool_map_boardered_type2, img_size))
        
        img = self.syn_img_final['img']
        img_with_mark = np.copy(img)
        
        # point1 = (self.syn_img_final['geo_feature_points'][0][1], self.syn_img_final['geo_feature_points'][0][0])
        # point2 = (self.syn_img_final['geo_feature_points'][1][1], self.syn_img_final['geo_feature_points'][1][0])
        # point3 = (self.syn_img_final['geo_feature_points'][2][1], self.syn_img_final['geo_feature_points'][2][0])
        # point4 = (self.syn_img_final['geo_feature_points'][3][1], self.syn_img_final['geo_feature_points'][3][0])
        # point5 = (self.syn_img_final['geo_feature_points'][4][1], self.syn_img_final['geo_feature_points'][4][0])
        # cv2.circle(img = img_with_mark,
        #            center = point1,
        #            radius = 10,
        #            color = (255,0,0),
        #            thickness = -1)
        # cv2.circle(img = img_with_mark,
        #            center = point2,
        #            radius = 10,
        #            color = (125,125,0),
        #            thickness = -1)
        # cv2.circle(img = img_with_mark,
        #            center = point3,
        #            radius = 10,
        #            color = (0,255,0),
        #            thickness = -1)
        # cv2.circle(img = img_with_mark,
        #            center = point4,
        #            radius = 10,
        #            color = (0,125,125),
        #            thickness = -1)
        # cv2.circle(img = img_with_mark,
        #            center = point5,
        #            radius = 10,
        #            color = (0,0,255),
        #            thickness = -1)
        
        # vector1 = self.syn_img_final['geo_feature_vectors'][0]
        # vector2 = self.syn_img_final['geo_feature_vectors'][1]
        # vector3 = self.syn_img_final['geo_feature_vectors'][2]
        # v1_p1 = point2
        # v1_p2 = (int(point2[0]+0.5*vector1[1]), int(point2[1]+0.5*vector1[0]))
        # v2_p1 = point4
        # v2_p2 = (int(point4[0]+0.5*vector2[1]), int(point4[1]+0.5*vector2[0]))
        # v3_p1 = point3
        # v3_p2 = (int(point3[0]+0.5*vector3[1]), int(point3[1]+0.5*vector3[0]))
        # cv2.arrowedLine(img = img_with_mark, 
        #             pt1 = v1_p1, 
        #             pt2 = v1_p2, 
        #             color = (255,125,0),
        #             thickness = 2)
        # cv2.arrowedLine(img = img_with_mark, 
        #             pt1 = v2_p1, 
        #             pt2 = v2_p2, 
        #             color = (0,125,255),
        #             thickness = 2)
        # cv2.arrowedLine(img = img_with_mark, 
        #             pt1 = v3_p1, 
        #             pt2 = v3_p2, 
        #             color = (125,255,125),
        #             thickness = 2)
        
        # tool_map = self.syn_img_final['tool_map']
        # tool_map = tool_map_rev2
        tool_map = self.syn_img_final['tool_map']
        if np.max(tool_map)>1:
            mask_green_channel = np.array((tool_map.astype(np.uint8)*120)).clip(0,255)
        else:
            mask_green_channel = np.array((tool_map.astype(np.uint8)*255)).clip(0,255)
        
        mask = np.zeros(img.shape, np.uint8)
        mask[:,:,1] = mask_green_channel
        
        mark_n_mask = np.concatenate((img, np.concatenate((img_with_mark, mask), axis=1)), axis = 1)
        cv2.line(mark_n_mask, pt1=(1000,0), pt2=(1000,1001), color=(255,255,255))
        cv2.line(mark_n_mask, pt1=(2000,0), pt2=(2000,1001), color=(255,255,255))
        
        cv2.imwrite(self.folder_img_n_gt + '/img_n_gt_' + str(self.syn_img_final['index']) + '.jpg', cv2.resize(mark_n_mask, (720, 240)))
        
        self.index = self.index + 1
        
        return None
    
    def show_final_result(self, show_img = True):
        if show_img == True:
            img_fn = self.syn_img_final['img']
            g_utils.show_img('result_syn_img', img_fn)
            
            img_with_mark = np.copy(img_fn)
            point1 = (self.syn_img_final['geo_feature_points'][0][1], self.syn_img_final['geo_feature_points'][0][0])
            point2 = (self.syn_img_final['geo_feature_points'][1][1], self.syn_img_final['geo_feature_points'][1][0])
            point3 = (self.syn_img_final['geo_feature_points'][2][1], self.syn_img_final['geo_feature_points'][2][0])
            point4 = (self.syn_img_final['geo_feature_points'][3][1], self.syn_img_final['geo_feature_points'][3][0])
            point5 = (self.syn_img_final['geo_feature_points'][4][1], self.syn_img_final['geo_feature_points'][4][0])
            cv2.circle(img = img_with_mark,
                       center = point1,
                       radius = 10,
                       color = (255,0,0),
                       thickness = -1)
            cv2.circle(img = img_with_mark,
                       center = point2,
                       radius = 10,
                       color = (125,125,0),
                       thickness = -1)
            cv2.circle(img = img_with_mark,
                       center = point3,
                       radius = 10,
                       color = (0,255,0),
                       thickness = -1)
            cv2.circle(img = img_with_mark,
                       center = point4,
                       radius = 10,
                       color = (0,125,125),
                       thickness = -1)
            cv2.circle(img = img_with_mark,
                       center = point5,
                       radius = 10,
                       color = (0,0,255),
                       thickness = -1)
            
            vector1 = self.syn_img_final['geo_feature_vectors'][0]
            vector2 = self.syn_img_final['geo_feature_vectors'][1]
            vector3 = self.syn_img_final['geo_feature_vectors'][2]
            v1_p1 = point2
            v1_p2 = (int(point2[0]+0.5*vector1[1]), int(point2[1]+0.5*vector1[0]))
            v2_p1 = point4
            v2_p2 = (int(point4[0]+0.5*vector2[1]), int(point4[1]+0.5*vector2[0]))
            v3_p1 = point3
            v3_p2 = (int(point3[0]+0.5*vector3[1]), int(point3[1]+0.5*vector3[0]))
            cv2.arrowedLine(img = img_with_mark, 
                        pt1 = v1_p1, 
                        pt2 = v1_p2, 
                        color = (255,125,0),
                        thickness = 2)
            cv2.arrowedLine(img = img_with_mark, 
                        pt1 = v2_p1, 
                        pt2 = v2_p2, 
                        color = (0,125,255),
                        thickness = 2)
            cv2.arrowedLine(img = img_with_mark, 
                        pt1 = v3_p1, 
                        pt2 = v3_p2, 
                        color = (125,255,125),
                        thickness = 2)
            
            g_utils.show_img('img_with_marks', img_with_mark)
            
            g_utils.show_tool_map('tool_map', self.syn_img_final['tool_map'])
            
        else:
            return None
        return None
    
    def save_data_frame(self, add_time_stamp = False, file_type = '.pkl'):
        if add_time_stamp == False:
            file_name = self.folder_dataframe + '/data_frame_syn_img' + file_type
        else:
            file_name = self.folder_dataframe + '/data_frame_syn_img' + str(int(time.time())) + file_type
        
        if file_type == '.pkl':
            self.df_syn_img_compressed.to_pickle(file_name, index = False, compression = 'xz')
        elif file_type == '.csv':
            self.df_syn_img_compressed.to_csv(file_name, index = False)
        return None
    
    def save_result_video(self, folder_path = 'video/video_0'):
        try:
            os.mkdir(folder_path)
        except:
            _= 1
        
        shape_df = self.df_syn_img_compressed.shape
        frame_num = shape_df[0]
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        raw_video_writer = cv2.VideoWriter(folder_path + '/raw_video.avi', fourcc, 20.0, (1001,1001))
        marked_video_writer = cv2.VideoWriter(folder_path + '/marked_video.avi', fourcc, 20.0, (1001,1001))
        mask_writer = cv2.VideoWriter(folder_path + '/mask_video.avi', fourcc, 20.0, (1001,1001))
        mark_mask_writer = cv2.VideoWriter(folder_path + '/mark_mask_video.avi', fourcc, 20.0, (2002,1001))
        
        for idx_frame in range(0, frame_num):
            cur_frame = self.df_syn_img_compressed.iloc[idx_frame]
            
            img = cur_frame['img']
            img_with_mark = np.copy(img)
            
            point1 = (cur_frame['geo_feature_points'][0][1], cur_frame['geo_feature_points'][0][0])
            point2 = (cur_frame['geo_feature_points'][1][1], cur_frame['geo_feature_points'][1][0])
            point3 = (cur_frame['geo_feature_points'][2][1], cur_frame['geo_feature_points'][2][0])
            point4 = (cur_frame['geo_feature_points'][3][1], cur_frame['geo_feature_points'][3][0])
            point5 = (cur_frame['geo_feature_points'][4][1], cur_frame['geo_feature_points'][4][0])
            cv2.circle(img = img_with_mark,
                       center = point1,
                       radius = 10,
                       color = (255,0,0),
                       thickness = -1)
            cv2.circle(img = img_with_mark,
                       center = point2,
                       radius = 10,
                       color = (125,125,0),
                       thickness = -1)
            cv2.circle(img = img_with_mark,
                       center = point3,
                       radius = 10,
                       color = (0,255,0),
                       thickness = -1)
            cv2.circle(img = img_with_mark,
                       center = point4,
                       radius = 10,
                       color = (0,125,125),
                       thickness = -1)
            cv2.circle(img = img_with_mark,
                       center = point5,
                       radius = 10,
                       color = (0,0,255),
                       thickness = -1)
            
            vector1 = cur_frame['geo_feature_vectors'][0]
            vector2 = cur_frame['geo_feature_vectors'][1]
            vector3 = cur_frame['geo_feature_vectors'][2]
            v1_p1 = point2
            v1_p2 = (int(point2[0]+0.5*vector1[1]), int(point2[1]+0.5*vector1[0]))
            v2_p1 = point4
            v2_p2 = (int(point4[0]+0.5*vector2[1]), int(point4[1]+0.5*vector2[0]))
            v3_p1 = point3
            v3_p2 = (int(point3[0]+0.5*vector3[1]), int(point3[1]+0.5*vector3[0]))
            cv2.arrowedLine(img = img_with_mark, 
                        pt1 = v1_p1, 
                        pt2 = v1_p2, 
                        color = (255,125,0),
                        thickness = 2)
            cv2.arrowedLine(img = img_with_mark, 
                        pt1 = v2_p1, 
                        pt2 = v2_p2, 
                        color = (0,125,255),
                        thickness = 2)
            cv2.arrowedLine(img = img_with_mark, 
                        pt1 = v3_p1, 
                        pt2 = v3_p2, 
                        color = (125,255,125),
                        thickness = 2)
            
            tool_map = cur_frame['tool_map']
            mask_green_channel = np.array((tool_map.astype(np.uint8)*255)).clip(0,255)
            mask = np.zeros(img.shape, np.uint8)
            mask[:,:,1] = mask_green_channel
            
            mark_n_mask = np.concatenate((img_with_mark, mask), axis=1)
            cv2.line(mark_n_mask, pt1=(1000,0), pt2=(1000,1001), color=(255,255,255))
            
            raw_video_writer.write(img)
            marked_video_writer.write(img_with_mark)
            mask_writer.write(mask)
            mark_mask_writer.write(mark_n_mask)
            
        raw_video_writer.release()
        marked_video_writer.release()
        mask_writer.release()
        mark_mask_writer.release()
        cv2.destroyAllWindows()
            
        return None
    
    
        
        

    
        