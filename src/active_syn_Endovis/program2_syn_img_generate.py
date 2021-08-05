# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 14:53:03 2020

@author: 75678
"""
import numpy as np
import cv2
import time
import random

import synthetic_img_generator as sig
import generator_utils as g_utils
from generator_utils import rand_from_range

random.seed(23)
np.random.seed(23)

source_folder = 'tool_bkgd_data_frame_12'

result_folder = 'data_frame_12_29_6'

show_img = False
synthetic_img_generator1 = sig.syn_img_generator()

synthetic_img_generator1.load_src_dataframes(df_tool_path = source_folder + '/data_frame_tool.pkl',
                                              df_background_path = source_folder + '/data_frame_background.pkl',
                                              small_df = True,
                                              source_path = source_folder,
                                              tool_types = None, 
                                              background_groups = [1,3,4],
                                              file_type = '.pkl',
                                              file_type_param = 'infer',
                                              diverse_bkgd_path = 'val2017')

synthetic_img_generator1.set_img_folder(folder_path = result_folder, add_time_stamp = False)

synthetic_img_generator1.init_syn_img_dataframe(start_index = 30000)

iteration_times = 10000
start_time = time.time()
for iteration in range (0,iteration_times):    

    flip = rand_from_range(-1, 1)
    alpha = rand_from_range(2000, 3000) # 1500
    sigma = rand_from_range(10, 15) # 15
    synthetic_img_generator1.select_tool(index = None,
                                         flip = flip,
                                         alpha = alpha, 
                                         sigma = sigma,
                                         use_random_shape = rand_from_range(-2, -1), 
                                         control_points = rand_from_range(3, 5), 
                                         rad = rand_from_range(0.1, 0.9), 
                                         edgy =rand_from_range(0.01, 0.9),
                                         show_img = show_img)
    synthetic_img_generator1.select_background(index = None,
                                               use_diverse_bkgd = 1,
                                               show_img = show_img)
    shrink_factor = g_utils.rand_from_range(0.9, 0.99)
    synthetic_img_generator1.shrink_tool(shrink_factor, show_img = show_img)
    
    x_factor = g_utils.rand_from_range(-0.07, 0.3)
    y_factor = g_utils.rand_from_range(-0.1, 0.1)
    r_factor = g_utils.rand_from_range(-45, 45)
    synthetic_img_generator1.move_tool(x_factor, 
                                       y_factor, 
                                       r_factor,
                                       show_img = show_img)
    
    refl_param = [rand_from_range(0.5,1.5),rand_from_range(0.1,0.9)] # 1, 0.5
    synthetic_img_generator1.render_reflection(index = None,
                                               refl_type = 2, 
                                               refl_param = refl_param,
                                               use_diverse_refl = rand_from_range(-1, 5), 
                                               diverse_bkgd_path = 'val2017',
                                               show_img = show_img)
    
    color_adjust_strength = rand_from_range(0.1, 0.4) # ((0.7, 1.0)
    brightness_adjust = rand_from_range(0.9, 1.3) #1.3
    fusion_blur = int(rand_from_range(10, 30))
    synthetic_img_generator1.combine_syn_img(color_adjust_strength = color_adjust_strength, 
                                             brightness_adjust = brightness_adjust,
                                             border_fusion = True,
                                             fusion_blur = fusion_blur,
                                             show_img = show_img)
    
    tool_shadow_range = rand_from_range(0, 10) # 0, 
    tool_shadow_brightness = rand_from_range(0.1, 0.4) # 0.2, 
    bkgd_shadow_range = rand_from_range(0, 20) # 15, 
    bkgd_shadow_brightness = rand_from_range(0.3,0.8) # 0.5,
    synthetic_img_generator1.render_shadow(rander_shadow = rand_from_range(-2, -1), 
                                           tool_shadow_range = tool_shadow_range, 
                                           tool_shadow_brightness = tool_shadow_brightness, 
                                           bkgd_shadow_range = bkgd_shadow_range, 
                                           bkgd_shadow_brightness = bkgd_shadow_brightness,
                                           show_img = show_img)
    
    # expo_lenth = int(1000 * shrink_factor),
    # expo_width = int(150 * shrink_factor),
    expo_taper = [rand_from_range(0, 0.3), rand_from_range(0.8, 1.4)]  # [0.1, 1.2],
    expo_strength = rand_from_range(0, 10) # 3, 
    expo_spread = 31 # 51,
    expo_angle = rand_from_range(0, 20) # 10, 
    expo_move = [rand_from_range(50, 200), rand_from_range(-20, 20)] # [120, 10],
    synthetic_img_generator1.render_exposure(render_exposure = rand_from_range(-2, -1),
                                             expo_lenth = int(1000 * shrink_factor),
                                             expo_width = int(150 * shrink_factor),
                                             expo_taper = expo_taper,
                                             expo_strength = expo_strength, 
                                             expo_spread = expo_spread,
                                             expo_angle = expo_angle, 
                                             expo_move = expo_move,
                                             show_img = show_img)
    
    border_center = (rand_from_range(115,125), rand_from_range(115,125))
    radius = rand_from_range(110,120) # 450
    ksize = (rand_from_range(1,3), rand_from_range(1,3)) # (15, 15)
    sig_x = rand_from_range(1,3)  # 27
    synthetic_img_generator1.set_bound_and_blur(border_center = border_center,
                                                radius = radius, 
                                                border_color = [[18.9, 5.5], [17.7, 5.3], [24.6, 7.1]],
                                                ksize = ksize,
                                                sig_x = sig_x,
                                                show_img = show_img)
    
    synthetic_img_generator1.elastic_dilation_erosion(elastic = rand_from_range(-4, 1), 
                                                      els_alpha = rand_from_range(2000, 3000), 
                                                      els_sigma = rand_from_range(10, 15), 
                                                      dila_ero = rand_from_range(-1.5, 1.5), 
                                                      val_type = 0, 
                                                      dila_ero_size = rand_from_range(1, 2),
                                                      img_elastic = rand_from_range(-4, 1), 
                                                      img_els_alpha = rand_from_range(2000, 3000), 
                                                      img_els_sigma = rand_from_range(10, 15))
    
    synthetic_img_generator1.add_to_dataframe(img_size = (240,240))
    
    print('Current Iteration:' + str(iteration))

end_time = time.time()
print('Time used:')
print(end_time - start_time)
synthetic_img_generator1.save_data_frame(add_time_stamp = False, file_type = '.csv')
