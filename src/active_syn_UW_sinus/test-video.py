import os
import numpy as np
import cv2
from model import *
from utils import *
import tensorflow as tf


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)
del_all_flags(tf.flags.FLAGS)
tf.reset_default_graph()




startFrm=150
#video read
vfile='../dataset/videoclips/Hanna_edit.mp4'
vr = cv2.VideoCapture(vfile)
fps = vr.get(cv2.CAP_PROP_FPS)
nfrm = vr.get(cv2.CAP_PROP_FRAME_COUNT)
sizev = (int(vr.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vr.get(cv2.CAP_PROP_FRAME_HEIGHT)))
size = (240,240)
#video write
vw = cv2.VideoWriter('hanna.avi',cv2.VideoWriter_fourcc('P','I','M','1'),int(fps),(size[0]*2,size[1]))


color_table = load_color_table('../dataset/labels.json')
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
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
          checkpoint_dir='./checkpoint/DeepLab_16_240_240',
          pretrain_dir='',
          train_dataset='../dataset/train',
          val_dataset='../dataset/train',
          num_class=2,
          color_table=color_table,is_train=False)
    if not net.load(net.checkpoint_dir)[0]:
        raise Exception("Cannot find checkpoint!")
        
    for i in range(1,int(nfrm)):
        retval,im=vr.read()
        if i<startFrm:
            continue
        
        im = im[:,40:-40,:]
        
        t0 = time.time()
        idxmap,colormap = net.inference(im)  
        t = time.time()-t0
    
        print("{}, time={}sec".format(i,t))
        
        
        colormap=cv2.cvtColor(colormap,cv2.COLOR_RGB2BGR)
        
        rst = np.hstack([im,colormap])
        cv2.imshow("s",rst)
        cv2.waitKey(1)
        vw.write(rst)
vw.release()