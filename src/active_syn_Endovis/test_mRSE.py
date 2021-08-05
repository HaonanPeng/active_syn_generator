import os
import numpy as np
import cv2
from model import *
from utils import *
import tensorflow as tf
from PIL import Image


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)
del_all_flags(tf.flags.FLAGS)
tf.reset_default_graph()


flags = tf.app.flags
flags.DEFINE_string("train_dataset", "../dataset/train", "train dataset direction")
flags.DEFINE_string("val_dataset", "../dataset/train", "val dataset direction")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/DeepLab_16_240_240", "checkpoint")
flags.DEFINE_string("img_dir", "../dataset/test/images", "img_dir")
flags.DEFINE_string("rst_dir", "./test-rsts", "rst_dir")
flags.DEFINE_string("gt_dir", "../dataset/test/labels", "gt_dir")
flags.DEFINE_string("rst_file", "tets_rst.txt", "gt_dir")
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
          input_width=240,
          input_height=240,
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
        stds_iou = []
        means_iou = []
        for i,file in enumerate(files):
            if not file.endswith(".jpg"):
                continue
            
            ious = []
            for i in range(6):
                img = Image.open(os.path.join(img_dir,file))
                img = img.rotate(i*60)
                img = np.asarray(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                gt = Image.open(os.path.join(gt_dir,file[:-4]+'.png'))
                gt = gt.rotate(i*60)
                gt = np.asarray(gt)
                        
            
                idxmap,colormap = net.inference(img)  
            
                gt_=1-gt
                #     
                output = cv2.resize(idxmap,(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_NEAREST)
                output_=1-output
            
                #
                if (np.count_nonzero(output)+np.count_nonzero(gt)) is 0:
                    iou = 1
                else:                        
                    iou = np.count_nonzero(gt*output)/(np.count_nonzero(output+gt)+0.000001)
                
                ious.append(iou)
            
            m_iou = np.mean(ious)
            std_iou = np.std(ious)
            stds_iou.append(std_iou)
            means_iou.append(m_iou)
        print("mean_iou_mean: {},  mean_iou_var: {}".format(np.mean(means_iou),np.mean(stds_iou)))
        
        file = open(FLAGS.rst_file, 'w')
        file.write("mean_iou_mean: {},  mean_iou_var: {}".format(np.mean(means_iou),np.mean(stds_iou)))
        file.close()