# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 09:51:13 2021

@author: 75678
"""

import cv2
import os
import shutil, errno
import numpy as np
import glob
import random

class active_learner():
    folder_workspace = None
    folder_learn_iter = []
    folder_labeled = []
    folder_unlabeled = []
    folder_test_set = None
    folder_query = []
    folder_query_added = []
    folder_checkpoint = []
    folder_background = []
    
    cur_iter = None
    total_images_number = None
    query_criterion = None
    
    current_query_value = []
    selected_query = []
    
    def create_folders(self, iteration):
        self.folder_learn_iter.append(self.folder_workspace + '/active_iter' + str(int(iteration)))
        self.folder_labeled.append(self.folder_learn_iter[iteration] + '/labeled')
        self.folder_unlabeled.append(self.folder_learn_iter[iteration] + '/unlabeled')
        create_folder(self.folder_learn_iter[iteration])
        create_folder(self.folder_labeled[iteration])
        create_folder(self.folder_labeled[iteration] + '/images')
        create_folder(self.folder_labeled[iteration] + '/labels')
        create_folder(self.folder_unlabeled[iteration])
        create_folder(self.folder_unlabeled[iteration] + '/images')
        create_folder(self.folder_unlabeled[iteration] + '/labels')
        
        self.folder_query.append(self.folder_learn_iter[iteration] + '/query')
        create_folder(self.folder_query[iteration])
        create_folder(self.folder_query[iteration] + '/images')
        create_folder(self.folder_query[iteration] + '/labels')
        
        self.folder_query_added.append(self.folder_learn_iter[iteration] + '/query_added')
        create_folder(self.folder_query_added[iteration])
        create_folder(self.folder_query_added[iteration] + '/images')
        create_folder(self.folder_query_added[iteration] + '/labels')
        
        self.folder_checkpoint.append(self.folder_learn_iter[iteration] + '/checkpoint')
        create_folder(self.folder_checkpoint[iteration])
        
        self.folder_background.append(self.folder_learn_iter[iteration] + '/background')
        create_folder(self.folder_background[iteration])
        return None
    
    def init_active_learner(self, proportion_test = 0.05, proportion_labeled = 0.1, query_criterion = 'entropy'):
        self.folder_learn_iter = []
        self.folder_labeled = []
        self.folder_unlabeled = []
        self.folder_query = []
        self.folder_query_added = []
        self.folder_checkpoint = []
        self.folder_background = []
        self.current_query_value = []
        self.selected_query = []
        
        self.query_criterion = query_criterion
        
        folder_source_images = self.folder_workspace + '/raw_data_base/images'
        folder_source_labels = self.folder_workspace + '/raw_data_base/labels'
        self.total_images_number = len(glob.glob(folder_source_images + '/' + '*.' + 'jpg'))
        
        self.folder_learn_iter.append(self.folder_workspace + '/active_iter0')
        self.folder_labeled.append(self.folder_workspace + '/active_iter0/labeled')
        self.folder_unlabeled.append(self.folder_workspace + '/active_iter0/unlabeled')
        create_folder(self.folder_learn_iter[0])
        create_folder(self.folder_labeled[0])
        create_folder(self.folder_labeled[0] + '/images')
        create_folder(self.folder_labeled[0] + '/labels')
        create_folder(self.folder_unlabeled[0])
        create_folder(self.folder_unlabeled[0] + '/images')
        create_folder(self.folder_unlabeled[0] + '/labels')
        
        self.folder_query.append(self.folder_workspace + '/active_iter0/query')
        create_folder(self.folder_query[0])
        create_folder(self.folder_query[0] + '/images')
        create_folder(self.folder_query[0] + '/labels')
        
        self.folder_query_added.append(self.folder_workspace + '/active_iter0/query_added')
        create_folder(self.folder_query_added[0])
        create_folder(self.folder_query_added[0] + '/images')
        create_folder(self.folder_query_added[0] + '/labels')
        
        self.folder_checkpoint.append(self.folder_workspace + '/active_iter0/checkpoint')
        create_folder(self.folder_checkpoint[0])
        
        self.folder_background.append(self.folder_workspace + '/active_iter0/background')
        create_folder(self.folder_background[0])
        copy_folder(source_folder = self.folder_workspace + '/init_background', target_folder = self.folder_background[0] , file_type = 'jpg')
        
        folder_unlabeled_images = self.folder_unlabeled[0] + '/images'
        folder_unlabeled_labels = self.folder_unlabeled[0] + '/labels'
        folder_labeled_images = self.folder_labeled[0] + '/images'
        folder_labeled_labels = self.folder_labeled[0] + '/labels'
        copy_folder(source_folder = folder_source_images, target_folder = folder_unlabeled_images , file_type = 'jpg')
        copy_folder(source_folder = folder_source_labels, target_folder = folder_unlabeled_labels , file_type = 'png')
        unlabeled_image_files_ = get_file_names(folder_unlabeled_images, 'jpg')
        labeled_image_files_ = random.sample(unlabeled_image_files_, int(proportion_labeled*len(unlabeled_image_files_)))
        for labeled_image_file in labeled_image_files_:
            labeled_label_file = labeled_image_file[:-3] + 'png'
            move_file(labeled_image_file, folder_unlabeled_images, folder_labeled_images)
            move_file(labeled_label_file, folder_unlabeled_labels, folder_labeled_labels)        
        
        self.folder_test_set = self.folder_workspace + '/test_set'
        create_folder(self.folder_test_set)
        folder_test_set_images = self.folder_test_set + '/images'
        folder_test_set_labels = self.folder_test_set + '/labels'
        create_folder(folder_test_set_images)
        create_folder(folder_test_set_labels)
        
        copy_folder2(self.folder_labeled[0], self.folder_query_added[0])
        
        test_set_size = int(proportion_test * self.total_images_number)   
        unlabeled_image_files_ = get_file_names(folder_unlabeled_images, 'jpg')
        test_image_files_ = random.sample(unlabeled_image_files_, test_set_size)
        for test_image_file in test_image_files_:
            test_label_file = test_image_file[:-3] + 'png'
            move_file(test_image_file, folder_unlabeled_images, folder_test_set_images)
            move_file(test_label_file, folder_unlabeled_labels, folder_test_set_labels)  
        return None
    
    def add_instance(self, img_pd, file_name, BALD_img_pd_ = None):
        if self.query_criterion == 'entropy':
            entropy = pixcel_wise_entropy(img_pd)
            self.current_query_value.append({'value' : entropy, 'file_name' : file_name})
        if self.query_criterion == 'MC_var':
            mean_var = np.mean(img_pd)
            self.current_query_value.append({'value' : mean_var, 'file_name' : file_name})
        if self.query_criterion == 'MVR':
            MVR = 1 - np.mean(np.max(img_pd, axis = 2))
            self.current_query_value.append({'value' : MVR, 'file_name' : file_name})
        if self.query_criterion == 'BALD':
            entropy = pixcel_wise_entropy(img_pd)
            BALD_entropy = np.zeros(entropy.shape)
            counter = 0
            for BALD_img_pd in BALD_img_pd_:
                counter = counter + 1
                BALD_entropy = BALD_entropy + pixcel_wise_entropy(BALD_img_pd)
            BALD_entropy = BALD_entropy/counter
            BALD = entropy - BALD_entropy
            self.current_query_value.append({'value' : BALD, 'file_name' : file_name})
                
        return None
    
    def sort_query_list(self):
        def temp_func(x):
            return x['value']
        if self.query_criterion == 'entropy' or self.query_criterion == 'MC_var' or self.query_criterion == 'BALD' or self.query_criterion == 'MVR':
            self.current_query_value.sort(reverse=True, key=temp_func)
        else:
            self.current_query_value.sort(reverse=False, key=temp_func)
        return None
    
    def show_query_list(self, head = 50):
        print('[Active Learner]: the maximum entropy select list:')
        print(self.current_query_value[:head])
        return
    
    def select_query(self, head_query = 0.05, skip = 3, random_query = 0.05):
        head_query = int(head_query * self.total_images_number)
        random_query = int(random_query * self.total_images_number)
        
        query_num = head_query * skip
        selected_query_new = []
        
        if random_query != 0:
            try:
                select_quey_random = random.sample(self.current_query_value, random_query)
            except:
                select_quey_random = self.current_query_value
                
            for query_sample in select_quey_random:
                selected_query_new.append(query_sample)
                img_file = query_sample['file_name']   
                copy_img_label(img_file, self.folder_unlabeled[self.cur_iter], self.folder_query[self.cur_iter])
                self.current_query_value.remove(query_sample)
        
        if head_query != 0:
            try:
                select_quey_head = self.current_query_value[:query_num]
            except:
                skip = 1
                try:
                    select_quey_head = self.current_query_value[:head_query]
                except:
                    select_quey_head = self.current_query_value
                    
            counter = 0
            for query_sample in select_quey_head:
                if (counter%skip) == 0:
                    selected_query_new.append(query_sample)
                    img_file = query_sample['file_name']   
                    copy_img_label(img_file, self.folder_unlabeled[self.cur_iter], self.folder_query[self.cur_iter])
                counter = counter + 1
        self.selected_query.append(selected_query_new)
        return None
    
    def inherit_from_previous_iter(self):
        self.current_query_value = []
        
        copy_folder2(self.folder_labeled[self.cur_iter-1], self.folder_labeled[self.cur_iter])
        copy_folder2(self.folder_unlabeled[self.cur_iter-1], self.folder_unlabeled[self.cur_iter])
        copy_folder(self.folder_background[self.cur_iter-1], self.folder_background[self.cur_iter], 'jpg')
        
        for query_sample in self.selected_query[self.cur_iter-1]:
            img_file = query_sample['file_name']
            move_img_label(img_file, self.folder_unlabeled[self.cur_iter], self.folder_query_added[self.cur_iter])
        
        return None   
        

def move_file(file_name, source_folder, target_folder):
    os.replace(source_folder + '/' + file_name, target_folder + '/' + file_name)
    return

def copy_file(file_name, source_folder, target_folder):
    shutil.copyfile(source_folder + '/' + file_name, target_folder + '/' + file_name)
    return

def copy_folder(source_folder, target_folder, file_type):
    file_names_ = get_file_names(source_folder, file_type)
    for file_name in file_names_:
        copy_file(file_name, source_folder, target_folder)
    return None

# copy the folder with '/images' and '/labels' sub folders
def copy_folder2(source_folder, target_folder):
    copy_folder(source_folder + '/images', target_folder + '/images', 'jpg')
    copy_folder(source_folder + '/labels', target_folder + '/labels', 'png')
    return None



# file_name should contain '.jpg', and the folders should not contain '/images' or '/labels'
def move_img_label(img_file, source_folder, target_folder):
    move_file(img_file, source_folder + '/images', target_folder + '/images')
    
    label_file = img_file[:-3] + 'png'
    move_file(label_file, source_folder + '/labels', target_folder + '/labels')
    return None

# file_name should contain '.jpg', and the folders should not contain '/images' or '/labels'
def copy_img_label(img_file, source_folder, target_folder):
    copy_file(img_file, source_folder + '/images', target_folder + '/images')
    
    label_file = img_file[:-3] + 'png'
    copy_file(label_file, source_folder + '/labels', target_folder + '/labels')
    return None
        
def get_file_names(folder_path, file_type):
    file_names = []
    img_files_paths = glob.glob(folder_path + '/' + '*.' + file_type)
    
    for img_files_path in img_files_paths:
        _head , img_file = os.path.split(img_files_path)
        file_names.append(img_file)
    
    return file_names



def create_folder(folder_path):
    try:
        os.mkdir(folder_path)
        return True
    except:
        _= 1
        return False

def delete_folder(source_folder):
    try:
        shutil.rmtree(source_folder)
        return True
    except:
        _= 1
        return False
    
    

def pixcel_wise_entropy(img_pd):
    entropy = 0.0
    
    for i in range(0, (img_pd.shape)[2]):
        class_pd = img_pd[:,:,i]
        class_pd[class_pd==0] = 1e-6
        # print(class_pd.shape)
        entropy = entropy - np.sum(class_pd * np.log(class_pd))
    
    return entropy

def show_prob_distribution(name, img_p):
    img0p = img_p[:,:,0]
    img1p = img_p[:,:,1]
    
    img0 = (240 * img0p).clip(0,255).astype(np.uint8)
    img1 = (240 * img1p).clip(0,255).astype(np.uint8)
    
    cv2.imshow(name + '0',img0)
    cv2.imshow(name + '1',img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None

def debug_print(string):
    print('||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
    print('||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
    print(string)
    print('||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
    print('||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
    
    return None

# def copy_folder(src, dst):
#     try:
#         shutil.copytree(src, dst)
#     except OSError as exc: # python >2.5
#         if exc.errno == errno.ENOTDIR:
#             shutil.copy(src, dst)
#         else: raise

    