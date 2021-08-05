from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import random
import cv2
from mobilenetv1 import *
from tensorflow.contrib import layers as contrib_layers
slim = tf.contrib.slim


try:
    from .ops import *
    from .utils import *
except:    
    from ops import *
    from utils import *

# [hyper paramters]-------------------------------------------------------
dropout_keep_prob = 1.0 # 1.0  original, this parameter seems make no affect on performance
keep_prob_train = 0.4 # 0.5 original, reducing this parameter can prevent overfitting
end_points_pointwise = 'Conv2d_13_pointwise' # 'Conv2d_13_pointwise' original, the depth of the encoder, if this is changed, the same parameter must be changed in 'mobilenetv1.py'
learning_rate = 0.001 # 0.0005  original, the learning rate should be dcreased accordingly to the increase of the training set. For example, if the training data is increased by 3 time, then then learning rate should be decreased by at least 3 times.
batch_norm_decay = 0.9 # 0.999 original, decrease this parameter can prevent overfitting
batch_norm_zero_debias_moving_mean = True # enable this can prevent overfitting
batch_norm_l2_regu = 0.1 # increase this can prevent overfitting
# [hyper paramters]-------------------------------------------------------

time_stamp = time.time()

class DeepLab(object):
  def __init__(self,sess,
          input_width,
          input_height,
          batch_size,
          img_pattern,
          label_pattern,
          checkpoint_dir,
          pretrain_dir,
          train_dataset,
          val_dataset,
          num_class,
          color_table,
          is_train=False):

    self.sess = sess
    self.is_train=is_train


    self.batch_size = batch_size
    self.num_class = num_class
    self.input_height = int(input_height)
    self.input_width = int(input_width)
    self.chn = 3

    self.learning_rate=learning_rate
    self.beta1=0.9
    self.seed=23

    self.model_name = "mobilenet_v1_1.0_224"

    self.train_dataset = train_dataset
    self.val_dataset = val_dataset
    self.img_pattern = img_pattern
    self.label_pattern = label_pattern
    self.checkpoint_dir = checkpoint_dir
    self.pretrain_dir = pretrain_dir
    self.color_table = color_table

    self.data = glob(os.path.join(self.train_dataset,"images", self.img_pattern))
    self.label = glob(os.path.join(self.train_dataset,"labels", self.label_pattern))
    self.val_data = glob(os.path.join(self.val_dataset,"images", self.img_pattern))
    self.val_label = glob(os.path.join(self.val_dataset,"labels", self.label_pattern))
    
    self.data.sort()
    self.label.sort()
    self.val_data.sort()
    self.val_label.sort()

    self.build_model()
    self.build_augmentation()
    
    
  def build_augmentation(self):
    image_dims = [self.input_height, self.input_width, self.chn]
    label_dims = [self.input_height, self.input_width,1]
        
    # augmentation modual
    self.im_raw = tf.placeholder(tf.float32,  image_dims, name='im_raw')
    self.label_raw = tf.placeholder(tf.int32, label_dims, name='label_raw')
    seed =23   
    def augment(image,color=False):
      r = image
      if color:
        r/=255.
        r = tf.image.random_hue(r,max_delta=0.1, seed=seed)
        r = tf.image.random_brightness(r,max_delta=0.3, seed=seed)
        r = tf.image.random_saturation(r,0.7,1.3, seed=seed)
        r = tf.image.random_contrast(r,0.7,1.3, seed=seed)
        r = tf.minimum(r, 1.0)
        r = tf.maximum(r, 0.0)
        r*=255.
      r = tf.image.random_flip_left_right(r, seed=seed)
      r = tf.image.random_flip_up_down(r,seed=seed)
      
      if color:
        r = tf.contrib.image.rotate(r,tf.random_uniform((), minval=-np.pi/180*180,maxval=np.pi/180*180,seed=seed),interpolation='BILINEAR')
      else:
        r = tf.contrib.image.rotate(r,tf.random_uniform((), minval=-np.pi/180*180,maxval=np.pi/180*180,seed=seed),interpolation='NEAREST')
     
      # resize
      r_width = tf.random_uniform([1],minval=int(0.7*self.input_width),maxval=int(1.3*self.input_width),dtype=tf.int32,seed=seed)
      r_height = tf.random_uniform([1],minval=int(0.7*self.input_height),maxval=int(1.3*self.input_height),dtype=tf.int32,seed=seed)
      
      if color:
          r = tf.image.resize_images(r,tf.concat([r_height,r_width],axis=0),method=tf.image.ResizeMethod.BILINEAR)
      else:
          r = tf.image.resize_images(r,tf.concat([r_height,r_width],axis=0),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      
      # pad if needed
      p_width = tf.reduce_max( tf.concat([r_width, tf.constant(self.input_width,shape=[1])],axis=0))
      p_height = tf.reduce_max(tf.concat([r_height,tf.constant(self.input_height,shape=[1])],axis=0))
      r = tf.image.resize_image_with_crop_or_pad(r, target_height=p_height, target_width=p_width)
      

#      # crop randomly
      
      dh = tf.cast(tf.random_uniform([0],minval=0,maxval=1,dtype=tf.float32,seed=seed)*tf.cast(p_height-self.input_height,dtype=tf.float32),dtype=tf.int32)
      dw = tf.cast(tf.random_uniform([0],minval=0,maxval=1,dtype=tf.float32,seed=seed)*tf.cast(p_width-self.input_width,dtype=tf.float32),dtype=tf.int32)
      r = tf.image.crop_to_bounding_box(r, offset_height = tf.reduce_sum(dh), offset_width = tf.reduce_sum(dw),
                                            target_height=self.input_height, target_width=self.input_width)
      
      
      return r
    
    self.im_aug,self.label_aug = [augment(self.im_raw,color=True),augment(self.label_raw)]
    

  def build_model(self):

    ##############################################################
    # inputs
    image_dims = [self.input_height, self.input_width, self.chn]
    label_dims = [self.input_height, self.input_width]

    self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='image')
    self.targets = tf.placeholder(tf.int32, [self.batch_size] + label_dims, name='label')
    self.keep_prob = tf.placeholder(tf.float32)
    
    ################################################################
    # layers
    layers = []

    h0 = self.inputs-127.5 #512
    
    n_rotate = 4
    hs = []
    for r in range(n_rotate):
        
        # rotate
        h = tf.contrib.image.rotate(h0, (np.pi/180)*(360*r/n_rotate),interpolation='NEAREST')        
        hs.append(h)
    h = tf.concat(hs,axis=0)
    
    
    end_points = {}
    with slim.arg_scope([slim.batch_norm], decay = batch_norm_decay, is_training=self.is_train, param_regularizers = contrib_layers.l2_regularizer(batch_norm_l2_regu), zero_debias_moving_mean = batch_norm_zero_debias_moving_mean):
#        _, end_points = resnet_v2_50(h,
#             num_classes=0,
#             is_training=self.is_train,
#             global_pool=False,
#             output_stride=16,
#             spatial_squeeze=False,
#             reuse=None,
#             scope=self.model_name)
        backbone_scope='MobilenetV1'
        _,end_points = mobilenet_v1(h,
                 num_classes=0,
                 dropout_keep_prob= dropout_keep_prob,
                 is_training=self.is_train,
                 min_depth=8,
                 depth_multiplier=1.0,
                 conv_defs=None,
                 prediction_fn=contrib_layers.softmax,
                 spatial_squeeze=False,
                 output_stride=16,
                 reuse=None,
                 scope=backbone_scope,
                 global_pool=False)
    
    h = end_points[end_points_pointwise]   
    skip = end_points['Conv2d_2_pointwise']   
        
        
    h = tf.nn.dropout(h,self.keep_prob,seed = self.seed)
    
    
    # atrous spatial pyramid pooling
    h = atrous_spatial_pyramid_pooling(h, output_stride=16, depth=256,is_train=self.is_train)
    # upsample*4
    h = tf.image.resize_bilinear(h, tf.shape(skip)[1:3])
    
    # skip connect low level features
    skip = conv2d(skip,32,ksize=1,stride=1,name="conv_skip") 
    skip = tf.nn.relu(batchnorm(skip,self.is_train,'bn_skip'))
    
    # concate and segment
    h = tf.concat([h,skip],axis=3)
    
    hs1 = tf.split(h,num_or_size_splits=n_rotate,axis=0)
    # un-rotate
    hs=[]
    ha=[]
    for r in range(n_rotate):
        h = tf.contrib.image.rotate(hs1[r], -(np.pi/180)*(360*r/n_rotate),interpolation='NEAREST')
        hs.append(h)
    
    h = tf.stack(hs,axis=-1)
    h = tf.reduce_mean(h,axis=-1)  
    
    h = separable_conv2d(h,128,ksize=3,name="conv_out1")
    h = tf.nn.relu(batchnorm(h,self.is_train,'bn_out1'))

    h = separable_conv2d(h,128,ksize=3,name="conv_out2")
    h = tf.nn.relu(h)
    
    h = separable_conv2d(h,self.num_class,ksize=3,name="conv_out3")
    # upsample
    h = tf.image.resize_bilinear(h, [self.input_height, self.input_width])
    ###########################################################################
    output_logits = h
    self.output_softmax = tf.nn.softmax(output_logits)   
    
    self.output = tf.cast(tf.argmax(self.output_softmax,axis=3),tf.uint8,name='outputs')
    
    
    ###########################################################################
    if self.is_train:
        #loss
        K=self.num_class
        label_map = tf.one_hot(tf.cast(self.targets,tf.int32),K)
        flat_label = tf.reshape(label_map,[-1,K])
        flat_out = tf.reshape(self.output_softmax,[-1,K])
        self.seg_loss = tf.reduce_mean(tf.multiply(-flat_label,tf.log(flat_out+0.000001)))
        
        
        self.loss_sum = scalar_summary("loss", self.seg_loss)
        self.val_loss_sum = scalar_summary("val_loss", self.seg_loss)
    # saver
    g_vars = tf.global_variables()
    bn_moving_vars = [g for g in g_vars if 'moving_' in g.name]
    self.tvars=tf.trainable_variables()
    self.load_vars = [var for var in self.tvars if self.pretrain_dir is '' or (backbone_scope in var.name and "biases" not in var.name)]
    self.saver = tf.train.Saver(self.tvars+bn_moving_vars,max_to_keep=1)
    if self.pretrain_dir is '':
        self.load_vars+=bn_moving_vars
    self.loader = tf.train.Saver(self.load_vars)
    
    
  def inference(self, img):
    shape0 = (img.shape[1],img.shape[0])
    img = cv2.resize(img,(self.input_width,self.input_height))
    
    
    inputs = np.array([img]).astype(np.float32)

    t0 = time.time()
    out_softmax = self.sess.run(self.output_softmax,feed_dict={self.inputs:inputs,self.keep_prob:1.0})
    t = time.time()-t0
    # print('infer_time: {}ms'.format(t*1000))    
    
    outp = out_softmax[0,:,:,:]
    
    out_put_pd = outp[:]
    
    thresh = 0.5
    outp[outp<thresh] = 0
    out = np.argmax(outp,axis=2)
    
    
    rst=idxmap2colormap(out,self.color_table)
    
    idxmap = cv2.resize(out,shape0,interpolation=cv2.INTER_NEAREST)
    colormap = cv2.resize(rst,shape0,interpolation=cv2.INTER_NEAREST)
    
    return idxmap, colormap, out_put_pd


      
      
  def train(self,config):
    loss_recorder = np.array([[0.5,0.5,0.5]])

    batch_num = len(self.data) // self.batch_size
    
    # learning rate
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = self.learning_rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           batch_num*config.epoch//4, 0.5, staircase=True)
      
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for updating moving average of batchnorm
    with tf.control_dependencies(update_ops):
      optim = tf.train.AdamOptimizer(learning_rate, beta1=self.beta1) \
              .minimize(self.seg_loss, var_list=self.tvars,global_step=global_step)
        
    
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.train_sum = merge_summary([self.loss_sum])
    self.writer = SummaryWriter(os.path.join(self.checkpoint_dir,"logs"), self.sess.graph)
  
    counter = 1
    start_time = time.time()
    if os.path.exists(self.pretrain_dir):
        could_load, checkpoint_counter = self.load_pretrain(self.pretrain_dir)
    else:
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")


    idxs = np.arange(len(self.data))
    idxv = np.arange(len(self.val_data))
    train_loss_queue = []
    for epoch in xrange(config.epoch):
      
      random.seed(self.seed)
      random.shuffle(idxs)
      random.seed(self.seed)
      random.shuffle(idxv)
      
      for idx in xrange(0, batch_num):
        file_idxs = idxs[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_images = [imread(self.data[i],resize_wh=(self.input_width,self.input_height),
                                     nearest_interpolate=True,grayscale=False) for i in file_idxs]
        batch_labels = [imread(self.label[i],resize_wh=(self.input_width,self.input_height),
                                     nearest_interpolate=True,grayscale=True) for i in file_idxs]
                     
        # augmentaion
        batch_images = [self.sess.run(self.im_aug,feed_dict={self.im_raw:im}) for im in batch_images]
        batch_labels = [self.sess.run(self.label_aug,feed_dict={self.label_raw:np.reshape(lb,[lb.shape[0],lb.shape[1],1])})[:,:,0] for lb in batch_labels]
        
        
        batch_images = np.array(batch_images).astype(np.float32)
        batch_labels = np.array(batch_labels).astype(np.int32)
        # Update gradient
        _,train_loss, summary_str,cur_lr = self.sess.run([optim,self.seg_loss, self.loss_sum,learning_rate],
                                                  feed_dict={ self.inputs: batch_images, self.targets: batch_labels,self.keep_prob: keep_prob_train})
        self.writer.add_summary(summary_str, counter)

        counter += 1
        train_loss_queue.append(train_loss)
        if len(train_loss_queue)>10:
            train_loss_queue.pop(0)
        train_loss_mean = np.mean(train_loss_queue)
        
        # print("Epoch[%2d/%2d] [%3d/%3d] time:%.2f min, loss:[%.4f], lr: %.5f" \
        #   % (epoch, config.epoch, idx, batch_num, (time.time() - start_time)/60, train_loss_mean, cur_lr))
        if counter% (batch_num//1) == 0:
          print('-------------------------------------------------------------')
          print('-------Epoch: ' + str(epoch) + '  ------Batch_bum: ' + str(idx))
          print('-------Train_Loss: ' + str(train_loss_mean))
          print('-------Learning_rate: ' + str(cur_lr))

        if counter% (batch_num//1) == 0:
          file_idx0 = 0#np.random.randint(len(self.val_data)-self.batch_size)
          file_idxs = idxv[file_idx0:self.batch_size+file_idx0]
          val_batch_images = [imread(self.val_data[i],resize_wh=(self.input_width,self.input_height),
                                     nearest_interpolate=True,grayscale=False) for i in file_idxs]
          val_batch_labels = [imread(self.val_label[i],resize_wh=(self.input_width,self.input_height),
                                     nearest_interpolate=True,grayscale=True) for i in file_idxs]
                        
          val_batch_images = np.array(val_batch_images).astype(np.float32)
          val_batch_labels = np.array(val_batch_labels).astype(np.int32)
          out, train_loss, summary_str = self.sess.run([self.output,self.seg_loss, self.val_loss_sum],
                                                  feed_dict={ self.inputs: val_batch_images, self.targets: val_batch_labels,self.keep_prob:1.0})
          self.writer.add_summary(summary_str, counter)
          
          ### -----------------------------------------------------------------
          
          print('-------Vali_Loss: ' + str(train_loss))
          ### -----------------------------------------------------------------
          
          disp_idx=(counter//(batch_num//1))%self.batch_size
          output=idxmap2colormap(out[disp_idx,:,:],self.color_table)
          label = idxmap2colormap(val_batch_labels[disp_idx,:,:],self.color_table)
          input = val_batch_images[disp_idx,:,:,:]
          rst=np.hstack((input,label,output))
          filename = "%08d_%05d_%05d.jpg" % (counter, 10000*train_loss_mean ,10000*train_loss)
          
          label=cv2.cvtColor(label,cv2.COLOR_RGB2BGR)
          output=cv2.cvtColor(output,cv2.COLOR_RGB2BGR)
          cv2.imwrite(os.path.join(self.checkpoint_dir,filename),rst)
          # loss_recorder = np.append(loss_recorder, [[counter, train_loss_mean, train_loss]], axis = 0)
          # np.savetxt('checkpoint/loss_record' + str(time_stamp) + '.txt', loss_recorder)
         
          
          
        if np.mod(counter, (batch_num//1)) == 0:
          self.save(self.checkpoint_dir, counter)    
    self.save(self.checkpoint_dir, counter)    

          
  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        'DeepLab', self.batch_size,
        self.input_height, self.input_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DeepLab.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    try:
      self.sess.run(tf.global_variables_initializer())
    except:
      self.sess.run(tf.initialize_all_variables().run())
    
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.loader.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
    
  def load_pretrain(self, pretrain_file):
    import re
    print(" [*] Reading checkpoints...")
    try:
      self.sess.run(tf.global_variables_initializer())
    except:
      self.sess.run(tf.initialize_all_variables().run())
    
    
    self.loader.restore(self.sess,os.path.join(self.pretrain_dir,self.model_name+".ckpt"))
#    tf.train.init_from_checkpoint(self.pretrain_file,{v.name.split(':')[0]:v for v in self.load_vars})
    counter = 0
    return True,counter
      
