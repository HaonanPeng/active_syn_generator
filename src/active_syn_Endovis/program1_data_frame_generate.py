# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 18:05:03 2020

@author: 75678
"""
import numpy as np
import random

import data_frame_generator as dfg

random.seed(23)
np.random.seed(23)

result_data_frame_folder = 'tool_bkgd_data_frame_15'

data_frame_generator1 = dfg.data_frame_generator()

data_frame_generator1.set_data_frame_folder(folder = result_data_frame_folder, using_small_df = True)

## [Initialization] if start building data frame from zero, use init functions---------
data_frame_generator1.init_tool_data_frame()
data_frame_generator1.init_background_data_frame()
## ------------------------------------------------------------------------------------

## [Load] if start with existed data frame, use load funtions--------------------------
# data_frame_generator1.load_tool_data_frame('data_frame/data_frame_tool_1600021122.pkl', file_type = '.pkl')
# data_frame_generator1.load_background_data_frame('data_frame/data_frame_background_1600021076.pkl', file_type = '.pkl')
## ------------------------------------------------------------------------------------

## [extract tools] from data base
data_frame_generator1.tool_extract(data_base_folder = 'sinus_data_base', 
                                   quantity = 100,
                                   extract_dilatation = 60)

## [Load backgrounds] This is a block to load background. Repeat this for multiple groups
data_frame_generator1.background_load_folder('backgrounds/Group1/', 
                                              group = 1, 
                                              img_type = 'jpg')
data_frame_generator1.background_load_folder('backgrounds/Group2/', 
                                              group = 2, 
                                              img_type = 'jpg')
data_frame_generator1.background_load_folder('backgrounds/Group3/', 
                                              group = 3, 
                                              img_type = 'jpg')
data_frame_generator1.background_load_folder('backgrounds/Group4/', 
                                              group = 4, 
                                              img_type = 'jpg')
# data_frame_generator1.background_load_folder('backgrounds/Group5/', 
#                                               group = 5, 
#                                               img_type = 'jpg')
# data_frame_generator1.background_load_folder('backgrounds/Group6/', 
#                                               group = 6, 
#                                               img_type = 'jpg')



data_frame_generator1.show_df_background_size()
data_frame_generator1.show_df_tool_size()

data_frame_generator1.save_data_frame_background(folder_path = result_data_frame_folder, add_time_stamp = False, file_type = '.pkl', file_type_param = 'infer')
data_frame_generator1.save_data_frame_tool(folder_path = result_data_frame_folder, add_time_stamp = False, file_type = '.pkl', file_type_param = 'infer')


# old tools --------------------------------------------------
# featurePoints2 = [[0,1000],
#                   [437,21],
#                   [500,0],
#                   [543,11],
#                   [1000,1000]]
# data_frame_generator1.tool_generate(texture_file_path = 'tools/selected_metal_texture_2.jpg', 
#                                     feature_points = featurePoints2, 
#                                     tool_type = 1)

# featurePoints3 = [[200,1000],
#                   [450,15],
#                   [500,0],
#                   [550,15],
#                   [800,1000]]
# data_frame_generator1.tool_generate(texture_file_path = 'tools/selected_metal_texture_3.jpg', 
#                                     feature_points = featurePoints3, 
#                                     tool_type = 1)

# featurePoints4 = [[300,1000],
#                   [430,30],
#                   [500,0],
#                   [550,20],
#                   [700,1000]]
# data_frame_generator1.tool_generate(texture_file_path = 'tools/selected_metal_texture_4.jpg', 
#                                     feature_points = featurePoints4, 
#                                     tool_type = 1)

# featurePoints5 = [[100,1000],
#                   [450,20],
#                   [490,0],
#                   [550,15],
#                   [920,1000]]
# data_frame_generator1.tool_generate(texture_file_path = 'tools/selected_metal_texture_5.jpg', 
#                                     feature_points = featurePoints5, 
#                                     tool_type = 1)

# featurePoints6 = [[0,1000],
#                   [410,30],
#                   [480,0],
#                   [550,25],
#                   [950,1000]]
# data_frame_generator1.tool_generate(texture_file_path = 'tools/selected_metal_texture_6.jpg', 
#                                     feature_points = featurePoints6, 
#                                     tool_type = 1)

# featurePoints7 = [[250,1000],
#                   [410,40],
#                   [500,0],
#                   [580,40],
#                   [800,1000]]
# data_frame_generator1.tool_generate(texture_file_path = 'tools/selected_metal_texture_7.jpg', 
#                                     feature_points = featurePoints7, 
#                                     tool_type = 1)

# featurePoints8 = [[50,1000],
#                   [450,25],
#                   [500,0],
#                   [590,38],
#                   [890,1000]]
# data_frame_generator1.tool_generate(texture_file_path = 'tools/selected_metal_texture_8.jpg', 
#                                     feature_points = featurePoints8, 
#                                     tool_type = 1)

# featurePoints9 = [[300,1000],
#                   [400,35],
#                   [500,0],
#                   [600,29],
#                   [750,1000]]
# data_frame_generator1.tool_generate(texture_file_path = 'tools/selected_metal_texture_9.jpg', 
#                                     feature_points = featurePoints9, 
#                                     tool_type = 1)

# featurePoints10 = [[210,1000],
#                   [450,50],
#                   [500,0],
#                   [560,50],
#                   [880,1000]]
# data_frame_generator1.tool_generate(texture_file_path = 'tools/selected_metal_texture_10.jpg', 
#                                     feature_points = featurePoints10, 
#                                     tool_type = 1)

# featurePoints11 = [[100,1000],
#                   [410,25],
#                   [500,0],
#                   [580,28],
#                   [900,1000]]
# data_frame_generator1.tool_generate(texture_file_path = 'tools/selected_metal_texture_11.jpg', 
#                                     feature_points = featurePoints11, 
#                                     tool_type = 1)

# featurePoints12 = [[100,1000],
#                   [350,150],
#                   [500,0],
#                   [650,150],
#                   [900,1000]]
# data_frame_generator1.tool_generate(texture_file_path = 'tools/selected_metal_texture_2.jpg', 
#                                     feature_points = featurePoints12, 
#                                     tool_type = 1)

# featurePoints13 = [[200,1000],
#                   [300,200],
#                   [440,0],
#                   [550,220],
#                   [800,1000]]
# data_frame_generator1.tool_generate(texture_file_path = 'tools/selected_metal_texture_3.jpg', 
#                                     feature_points = featurePoints13, 
#                                     tool_type = 1)

# featurePoints14 = [[100,1000],
#                   [380,180],
#                   [500,0],
#                   [600,160],
#                   [950,1000]]
# data_frame_generator1.tool_generate(texture_file_path = 'tools/selected_metal_texture_4.jpg', 
#                                     feature_points = featurePoints14, 
#                                     tool_type = 1)

# featurePoints15 = [[150,1000],
#                   [400,100],
#                   [490,0],
#                   [600,100],
#                   [860,1000]]
# data_frame_generator1.tool_generate(texture_file_path = 'tools/selected_metal_texture_5.jpg', 
#                                     feature_points = featurePoints15, 
#                                     tool_type = 1)

# featurePoints16 = [[0,1000],
#                   [400,200],
#                   [500,0],
#                   [600,210],
#                   [950,1000]]
# data_frame_generator1.tool_generate(texture_file_path = 'tools/selected_metal_texture_6.jpg', 
#                                     feature_points = featurePoints16, 
#                                     tool_type = 1)

# featurePoints17 = [[250,1000],
#                   [310,190],
#                   [500,0],
#                   [650,160],
#                   [800,1000]]
# data_frame_generator1.tool_generate(texture_file_path = 'tools/selected_metal_texture_7.jpg', 
#                                     feature_points = featurePoints17, 
#                                     tool_type = 1)

# featurePoints18 = [[120,1000],
#                   [450,80],
#                   [500,0],
#                   [550,100],
#                   [950,1000]]
# data_frame_generator1.tool_generate(texture_file_path = 'tools/selected_metal_texture_8.jpg', 
#                                     feature_points = featurePoints18, 
#                                     tool_type = 1)

# featurePoints19 = [[300,1000],
#                   [400,300],
#                   [500,0],
#                   [600,280],
#                   [750,1000]]
# data_frame_generator1.tool_generate(texture_file_path = 'tools/selected_metal_texture_9.jpg', 
#                                     feature_points = featurePoints19, 
#                                     tool_type = 1)

# featurePoints20 = [[230,1000],
#                   [450,110],
#                   [500,0],
#                   [560,130],
#                   [890,1000]]
# data_frame_generator1.tool_generate(texture_file_path = 'tools/selected_metal_texture_10.jpg', 
#                                     feature_points = featurePoints20, 
#                                     tool_type = 1)