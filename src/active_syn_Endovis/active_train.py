# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 12:59:31 2021

@author: Haonan
"""

import os
import numpy as np
import cv2
from model import *
from utils import *
import tensorflow as tf
import time

from PIL import Image
import active_learner as al
import active_syn_generator as asg
from generator_utils import rand_from_range

folder_workspace = './active_learning_try19'
active_iterations = 3


# total_training_strenth = 30 * 3000.0
# max_epoch = 2000

proportion_labeled = 0.05
proportion_test = 0.0

head_query_per_iter = 0.02
query_skip = 15
random_query_per_iter = 0.02
query_per_iter = head_query_per_iter + random_query_per_iter

# epoch = int(30/(proportion_labeled + (active_iterations-1)*query_per_iter))
epoch = 2

syn1_per_query = 1
syn2_per_query = 1

extract_dilation = 15

img_size = (427,240)
batch_size = 4
check_point_folder_name = '/DeepLab_' + str(batch_size) + '_' + str(img_size[1]) + '_' + str(img_size[0])
# '/DeepLab_16_427_240'

show_img = False # for debug, not working for ubuntu

random_seed = 24
tf.set_random_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)

start_time = time.time()

# epoch_iteration = np.zeros(active_iterations, np.int32)
# for i in range(0, active_iterations):
#     # epoch_iteration[i] = np.min((total_training_strenth/(3000*proportion_labeled + i*(1+syn1_per_query+syn2_per_query)*query_per_iter), max_epoch))
#     epoch_iteration[i] = np.min((total_training_strenth/(3000*proportion_labeled + i*query_per_iter), max_epoch))

active_learner1 = al.active_learner()
active_learner1.folder_workspace = folder_workspace
active_learner1.init_active_learner(proportion_test = proportion_test, 
                                    proportion_labeled = proportion_labeled, 
                                    query_criterion = 'entropy')

active_syn_generator1 = asg.active_syn_generator()
active_syn_generator1.init_syn_img_generator(folder_learn_iter = active_learner1.folder_learn_iter[0],
                                             extract_dilation = extract_dilation, 
                                             img_size = img_size,
                                             init_bkgd = False, 
                                             generate_bkgd = True,
                                             output_process = False) # output process is only usable with 1 iteration

vali_evaluation_result = []
test_evaluation_result = []

for a_iter in range(0, active_iterations):
    active_learner1.cur_iter = a_iter
    
    if a_iter==0:
        active_syn_generator1.generate_background()
        
        active_syn_generator1.load_tool_and_bkgd()
        query_img_list_ = al.get_file_names(active_learner1.folder_query_added[a_iter] + '/images', 'jpg')
        
        for query_img_file in query_img_list_:
            query_label_file = query_img_file[:-3] + 'png'
            
            image = cv2.imread(active_learner1.folder_query_added[a_iter] + '/images/' + query_img_file)
            label = cv2.imread(active_learner1.folder_query_added[a_iter] + '/labels/' + query_label_file, cv2.IMREAD_GRAYSCALE)
            
            if np.sum(label) < 40: # if the query image has no tool or only a small tool
                continue
            active_syn_generator1.file_name = query_img_file[:-4]
            
            syn1_images_, syn1_labels_, syn2_images_, syn2_labels_ = active_syn_generator1.generate_syn_img(image, label, dilation = extract_dilation,
                                                                                                            syn1 = syn1_per_query, syn2 = syn2_per_query, multi_gen = 2,                                                                                           
                                                                                flip=(0.1, 0.8), 
                                                                                shrink_factor=(0.9, 1.2), 
                                                                                 x_factor=(-0.1, 0.1), 
                                                                                 y_factor=(-0.1, 0.1), 
                                                                                 r_factor=(-10, 10), 
                                                                                 color_adjust_strength=(0.4, 1.0), 
                                                                                 brightness_adjust=(0.9, 1.3), 
                                                                                 fusion_blur=(20, 40), 
                                                                                 border_center=((115,125), (115,125)), 
                                                                                 radius=(150,170), 
                                                                                 ksize=(1,3), 
                                                                                 sig_x=(1,3),
                                                                                 elastic= (-2, -1), 
                                                                                 els_alpha=(2000, 3000), 
                                                                                 els_sigma=(10, 15), 
                                                                                 dila_ero=(-0.5, 0.5), 
                                                                                 val_type=0, 
                                                                                 dila_ero_size=(1, 2), 
                                                                                 img_elastic=(-4, -1), 
                                                                                 img_els_alpha=(2000, 3000), 
                                                                                 img_els_sigma=(10, 15),
                                                                                 img_size = img_size,
                                                                                 show_img = show_img)
            al.create_folder(active_learner1.folder_query_added[a_iter] + '/img_n_gt')
            count = 0
            for syn1_image in syn1_images_:
                syn1_label = syn1_labels_[count]
                cv2.imwrite(active_learner1.folder_query_added[a_iter] + '/images/' + query_img_file[:-4] + '_syn1_' + str(count) + '.jpg', syn1_image)
                cv2.imwrite(active_learner1.folder_query_added[a_iter] + '/labels/' + query_img_file[:-4] + '_syn1_' + str(count) + '.png', syn1_label)
                count = count + 1
                
                ground_truth = np.zeros(syn1_image.shape)
                ground_truth[:,:,1] = (syn1_label * 250).clip(0,255).astype(np.uint8)
                img_n_gt = np.hstack((syn1_image, ground_truth))
                cv2.imwrite(active_learner1.folder_query_added[a_iter] + '/img_n_gt/' + query_img_file[:-4] + '_syn1_' + str(count) + '.jpg', img_n_gt)
                
            count = 0
            for syn2_image in syn2_images_:
                syn2_label = syn2_labels_[count]
                cv2.imwrite(active_learner1.folder_query_added[a_iter] + '/images/' + query_img_file[:-4] + '_syn2_' + str(count) + '.jpg', syn2_image)
                cv2.imwrite(active_learner1.folder_query_added[a_iter] + '/labels/' + query_img_file[:-4] + '_syn2_' + str(count) + '.png', syn2_label)
                count = count + 1
                
                ground_truth = np.zeros(syn2_image.shape)
                ground_truth[:,:,1] = (syn2_label * 250).clip(0,255).astype(np.uint8)
                img_n_gt = np.hstack((syn2_image, ground_truth))
                cv2.imwrite(active_learner1.folder_query_added[a_iter] + '/img_n_gt/' + query_img_file[:-4] + '_syn2_' + str(count) + '.jpg', img_n_gt)
        al.copy_folder2(active_learner1.folder_query_added[a_iter], active_learner1.folder_labeled[a_iter])
        
        
    if a_iter>0:
        active_learner1.create_folders(iteration = a_iter)
        active_learner1.inherit_from_previous_iter()
        
        active_syn_generator1.init_new_iteration(iteration = a_iter, 
                                                 folder_learn_iter = active_learner1.folder_learn_iter[a_iter], 
                                                 dilation=extract_dilation)
        
        active_syn_generator1.load_tool_and_bkgd()
        
        query_img_list_ = al.get_file_names(active_learner1.folder_query_added[a_iter] + '/images', 'jpg')
        
        for query_img_file in query_img_list_:
            query_label_file = query_img_file[:-3] + 'png'
            
            image = cv2.imread(active_learner1.folder_query_added[a_iter] + '/images/' + query_img_file)
            label = cv2.imread(active_learner1.folder_query_added[a_iter] + '/labels/' + query_label_file, cv2.IMREAD_GRAYSCALE)
            
            if np.sum(label) < 40: # if the query image has no tool or only a small tool
                continue
        
            syn1_images_, syn1_labels_, syn2_images_, syn2_labels_ = active_syn_generator1.generate_syn_img(image, label, dilation = extract_dilation,
                                                                                                            syn1 = syn1_per_query, syn2 = syn2_per_query, multi_gen = 2,                                                                                           
                                                                                flip=(0.1, 0.8), 
                                                                                shrink_factor=(0.9, 1.2), 
                                                                                 x_factor=(-0.1, 0.1), 
                                                                                 y_factor=(-0.1, 0.1), 
                                                                                 r_factor=(-10, 10), 
                                                                                 color_adjust_strength=(0.4, 1.0), 
                                                                                 brightness_adjust=(0.9, 1.3), 
                                                                                 fusion_blur=(20, 40), 
                                                                                 border_center=((115,125), (115,125)), 
                                                                                 radius=(150,170), 
                                                                                 ksize=(1,3), 
                                                                                 sig_x=(1,3),
                                                                                 elastic= (-2, -1), 
                                                                                 els_alpha=(2000, 3000), 
                                                                                 els_sigma=(10, 15), 
                                                                                 dila_ero=(-0.5, 0.5), 
                                                                                 val_type=0, 
                                                                                 dila_ero_size=(1, 2), 
                                                                                 img_elastic=(-4, -1), 
                                                                                 img_els_alpha=(2000, 3000), 
                                                                                 img_els_sigma=(10, 15),
                                                                                 img_size = img_size,
                                                                                 show_img = show_img)
            al.create_folder(active_learner1.folder_query_added[a_iter] + '/img_n_gt')
            count = 0
            for syn1_image in syn1_images_:
                syn1_label = syn1_labels_[count]
                cv2.imwrite(active_learner1.folder_query_added[a_iter] + '/images/' + query_img_file[:-4] + '_syn1_' + str(count) + '.jpg', syn1_image)
                cv2.imwrite(active_learner1.folder_query_added[a_iter] + '/labels/' + query_img_file[:-4] + '_syn1_' + str(count) + '.png', syn1_label)
                count = count + 1
                
                ground_truth = np.zeros(syn1_image.shape)
                ground_truth[:,:,1] = (syn1_label * 250).clip(0,255).astype(np.uint8)
                img_n_gt = np.hstack((syn1_image, ground_truth))
                cv2.imwrite(active_learner1.folder_query_added[a_iter] + '/img_n_gt/' + query_img_file[:-4] + '_syn1_' + str(count) + '.jpg', img_n_gt)
                
            count = 0
            for syn2_image in syn2_images_:
                syn2_label = syn2_labels_[count]
                cv2.imwrite(active_learner1.folder_query_added[a_iter] + '/images/' + query_img_file[:-4] + '_syn2_' + str(count) + '.jpg', syn2_image)
                cv2.imwrite(active_learner1.folder_query_added[a_iter] + '/labels/' + query_img_file[:-4] + '_syn2_' + str(count) + '.png', syn2_label)
                count = count + 1
                
                ground_truth = np.zeros(syn2_image.shape)
                ground_truth[:,:,1] = (syn2_label * 250).clip(0,255).astype(np.uint8)
                img_n_gt = np.hstack((syn2_image, ground_truth))
                cv2.imwrite(active_learner1.folder_query_added[a_iter] + '/img_n_gt/' + query_img_file[:-4] + '_syn2_' + str(count) + '.jpg', img_n_gt)
        al.copy_folder2(active_learner1.folder_query_added[a_iter], active_learner1.folder_labeled[a_iter])
                           
# Training CNN -----------------------------------------------------------------------------------
    def del_all_flags(FLAGS):
        flags_dict = FLAGS._flags()
        keys_list = [keys for keys in flags_dict]
        for keys in keys_list:
            FLAGS.__delattr__(keys)
    del_all_flags(tf.flags.FLAGS)
    tf.reset_default_graph()
    tf.set_random_seed(23)
    
    flags = tf.app.flags
    # epoch = int(epoch_iteration[a_iter])
    print('||||||||||||||||||||||||||||||||||')
    print('current epoch:')
    print(str(epoch))
    print('||||||||||||||||||||||||||||||||||')
    
    flags.DEFINE_integer("epoch", epoch, "Epoch to train [25]")
    
    # flags.DEFINE_integer("epoch",epoch, "Epoch to train [25]")
    flags.DEFINE_integer("batch_size", batch_size, "The size of batch images [64]")
    flags.DEFINE_integer("input_height", img_size[1], "The size of image to use (will be center cropped). [108]")
    flags.DEFINE_integer("input_width", img_size[0], "The size of image to use (will be center cropped). If None, same value as input_height [None]")
    
    # the folder of training set and validation set, must contain a sub-folder called 'images' and a sub-folder called 'labels' 
    flags.DEFINE_string("train_dataset", active_learner1.folder_labeled[a_iter], "train dataset direction")
    flags.DEFINE_string("val_dataset", active_learner1.folder_unlabeled[a_iter], "train dataset direction")
    
    flags.DEFINE_string("img_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
    flags.DEFINE_string("label_pattern", "*.png", "Glob pattern of filename of input labels [*]")
    flags.DEFINE_string("checkpoint_dir", active_learner1.folder_checkpoint[a_iter], "Directory name to save the checkpoints [checkpoint]")
    flags.DEFINE_string("pretrain_dir", "./mobilenet_v1_1.0_224", "")
    flags.DEFINE_string("gpu", '0', "gpu")
    FLAGS = flags.FLAGS
    
    pp.pprint(flags.FLAGS.__flags)
    
    if not os.path.exists(FLAGS.checkpoint_dir):
      os.makedirs(FLAGS.checkpoint_dir)
      
    color_table = load_color_table('./labels.json')
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True
    tf.reset_default_graph()
    with tf.Session(config=run_config) as sess:
      
      net = DeepLab(
            sess,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            batch_size=FLAGS.batch_size,
            img_pattern=FLAGS.img_pattern,
            label_pattern=FLAGS.label_pattern,
            checkpoint_dir=FLAGS.checkpoint_dir,
            pretrain_dir=FLAGS.pretrain_dir,
            train_dataset=FLAGS.train_dataset,
            val_dataset=FLAGS.val_dataset,
            num_class=2,
            color_table=color_table,is_train=True)
      
      net.train(FLAGS)
        
# Evaluating CNN and query labeling using validation (unlabeled data)-----------------------------------------------------------------------
    def del_all_flags(FLAGS):
        flags_dict = FLAGS._flags()
        keys_list = [keys for keys in flags_dict]
        for keys in keys_list:
            FLAGS.__delattr__(keys)
    del_all_flags(tf.flags.FLAGS)
    tf.reset_default_graph()
    
    
    flags = tf.app.flags
    flags.DEFINE_string("train_dataset", active_learner1.folder_labeled[a_iter], "train dataset direction")
    flags.DEFINE_string("val_dataset", active_learner1.folder_unlabeled[a_iter], "val dataset direction")
    flags.DEFINE_string("checkpoint_dir", active_learner1.folder_checkpoint[a_iter] + check_point_folder_name, "checkpoint")
    flags.DEFINE_string("img_dir", active_learner1.folder_unlabeled[a_iter] + '/images', "img_dir")
    flags.DEFINE_string("rst_dir", active_learner1.folder_learn_iter[a_iter] + "/vali-rsts", "rst_dir")
    flags.DEFINE_string("gt_dir", active_learner1.folder_unlabeled[a_iter] + '/labels', "gt_dir")
    flags.DEFINE_string("rst_file", active_learner1.folder_learn_iter[a_iter] + "_vali_rst.txt", "gt_dir")
    flags.DEFINE_string("gpu", '0', "gpu")
    FLAGS = flags.FLAGS
    
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    
    test_all=True
    color_table = load_color_table('./labels.json')
    run_config = tf.ConfigProto()
    sess=tf.Session(config=run_config)
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        net = DeepLab(
              sess,
              input_width=img_size[0],
              input_height=img_size[1],
              batch_size=1,
              img_pattern="*.jpg",
              label_pattern="*.png",
              checkpoint_dir=FLAGS.checkpoint_dir,
              pretrain_dir='',
              train_dataset=FLAGS.train_dataset,
              val_dataset=FLAGS.val_dataset,
              num_class=2,
              color_table=color_table,is_train=False)
        if not net.load(net.checkpoint_dir)[0]:
            raise Exception("Cannot find checkpoint!")
            
        if test_all:
        
            #test on train
            img_dir = FLAGS.img_dir
            rst_dir = FLAGS.rst_dir
            gt_dir = FLAGS.gt_dir
            
            if not os.path.exists(rst_dir):
                os.makedirs(rst_dir)
    
            
            files=os.listdir(img_dir)
            for i,file in enumerate(files):
                if not file.endswith(".jpg"):
                    continue
                
                
                img = cv2.imread(os.path.join(img_dir,file))
                # print(img_dir)
                # print(file)
                
                idxmap, colormap, out_put_pd = net.inference(img)  
                
                active_learner1.add_instance(img_pd = out_put_pd, file_name = file)
                # al.show_prob_distribution('image_porb', out_put_pd)
                
                colormap=cv2.cvtColor(colormap,cv2.COLOR_RGB2BGR)
                
                cv2.imwrite(os.path.join(rst_dir,file[:-4]+'.png'),colormap) 
                
            mean_dice, mean_iou = evaluate_seg_result(rst_dir, gt_dir, FLAGS.rst_file)
            
            vali_evaluation_result.append([mean_dice, mean_iou])
            
            active_learner1.sort_query_list()
            active_learner1.show_query_list(head = 10)
            active_learner1.select_query(head_query = head_query_per_iter, skip = query_skip, random_query = random_query_per_iter)
            
# Evaluating CNN and using test set-----------------------------------------------------------------------
    def del_all_flags(FLAGS):
        flags_dict = FLAGS._flags()
        keys_list = [keys for keys in flags_dict]
        for keys in keys_list:
            FLAGS.__delattr__(keys)
    del_all_flags(tf.flags.FLAGS)
    tf.reset_default_graph()
    
    
    flags = tf.app.flags
    flags.DEFINE_string("train_dataset", active_learner1.folder_labeled[a_iter], "train dataset direction")
    flags.DEFINE_string("val_dataset", active_learner1.folder_unlabeled[a_iter], "val dataset direction")
    flags.DEFINE_string("checkpoint_dir", active_learner1.folder_checkpoint[a_iter] + check_point_folder_name, "checkpoint")
    flags.DEFINE_string("img_dir", active_learner1.folder_test_set + '/images', "img_dir")
    flags.DEFINE_string("rst_dir", active_learner1.folder_learn_iter[a_iter] + "/test-rsts", "rst_dir")
    flags.DEFINE_string("gt_dir", active_learner1.folder_test_set + '/labels', "gt_dir")
    flags.DEFINE_string("rst_file", active_learner1.folder_learn_iter[a_iter] + "_test_rst.txt", "gt_dir")
    flags.DEFINE_string("gpu", '0', "gpu")
    FLAGS = flags.FLAGS
    
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    
    test_all=True
    color_table = load_color_table('./labels.json')
    run_config = tf.ConfigProto()
    sess=tf.Session(config=run_config)
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        net = DeepLab(
              sess,
              input_width=img_size[0],
              input_height=img_size[1],
              batch_size=1,
              img_pattern="*.jpg",
              label_pattern="*.png",
              checkpoint_dir=FLAGS.checkpoint_dir,
              pretrain_dir='',
              train_dataset=FLAGS.train_dataset,
              val_dataset=FLAGS.val_dataset,
              num_class=2,
              color_table=color_table,is_train=False)
        if not net.load(net.checkpoint_dir)[0]:
            raise Exception("Cannot find checkpoint!")
            
        if test_all:
        
            #test on train
            img_dir = FLAGS.img_dir
            rst_dir = FLAGS.rst_dir
            gt_dir = FLAGS.gt_dir
            
            if not os.path.exists(rst_dir):
                os.makedirs(rst_dir)
    
            
            files=os.listdir(img_dir)
            for i,file in enumerate(files):
                if not file.endswith(".jpg"):
                    continue
                
                
                img = cv2.imread(os.path.join(img_dir,file))
                # print(img_dir)
                # print(file)
                
                idxmap, colormap, out_put_pd = net.inference(img)  
                
                # active_learner1.add_instance(img_pd = out_put_pd, file_name = file)
                # al.show_prob_distribution('image_porb', out_put_pd)
                
                colormap=cv2.cvtColor(colormap,cv2.COLOR_RGB2BGR)
                
                cv2.imwrite(os.path.join(rst_dir,file[:-4]+'.png'),colormap) 
                
            mean_dice, mean_iou = evaluate_seg_result(rst_dir, gt_dir, FLAGS.rst_file)
            
            test_evaluation_result.append([mean_dice, mean_iou])


    print('[Active Learner]: Active learning interation complete, completed iteration:' + str(a_iter))
    
print('Active training ended, validation result:')
print(vali_evaluation_result)
print('Active training ended, test result:')
print(test_evaluation_result)

print('Total time cost(second):')
print(str(time.time()-start_time))