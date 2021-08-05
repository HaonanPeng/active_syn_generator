import os
import numpy as np
import cv2
from model import *
from utils import *
import tensorflow as tf
from PIL import Image
import active_learner as al


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)
del_all_flags(tf.flags.FLAGS)
tf.reset_default_graph()


flags = tf.app.flags
flags.DEFINE_string("train_dataset", "D:/mafa_deeplab_Shan/code/uw-sinus-surgery-CL/cadaver", "train dataset direction")
flags.DEFINE_string("val_dataset", "D:/mafa_deeplab_Shan/code/uw-sinus-surgery-CL/cadaver", "val dataset direction")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/df_12_29_5_try4/checkpoint", "checkpoint")
flags.DEFINE_string("img_dir", "D:/mafa_deeplab_Shan/code/uw-sinus-surgery-CL/cadaver_test/images", "img_dir")
flags.DEFINE_string("rst_dir", "./test-rsts", "rst_dir")
flags.DEFINE_string("gt_dir", "D:/mafa_deeplab_Shan/code/uw-sinus-surgery-CL/cadaver_test/labels", "gt_dir")
flags.DEFINE_string("rst_file", "tets_rst.txt", "gt_dir")
flags.DEFINE_string("gpu", '0', "gpu")
FLAGS = flags.FLAGS


os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
def main(_):
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
            active_learner1 = al.active_learner()
        
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
                
                active_learner1.add_instance(img_pd = out_put_pd, file_name = file, criterion = 'entropy')
                # al.show_prob_distribution('image_porb', out_put_pd)
                
                colormap=cv2.cvtColor(colormap,cv2.COLOR_RGB2BGR)
                
                cv2.imwrite(os.path.join(rst_dir,file[:-4]+'.png'),colormap) 
                
            evaluate_seg_result(rst_dir, gt_dir, FLAGS.rst_file)
            
            active_learner1.sort_select_list(type = 'descending')
            active_learner1.show_select_list(criterion = 'entropy')
        
        else:
            for i in range(12):
                img_dir = "../dataset/test/images"
                file="hanna21240.jpg"
                
                img = Image.open(os.path.join(img_dir,file))
                img = img.rotate(i*30)
                img = np.asarray(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
    #            h0,w0,c0 = img.shape
    #            img = cv2.resize(img, (w0*3//3,h0*3//3))
    #            padh = (h0-img.shape[1])//2
    #            padw = (w0-img.shape[0])//2
    #            img = np.pad(img,((padh,padh),(padw,padw),(0,0)),mode='constant',constant_values=0)
                
                idxmap,colormap = net.inference(img) 
                
                rst = np.hstack([img,colormap])
                cv2.imwrite('/home/user/'+file[:-4]+'_dl_rot{}.png'.format(i*30),rst)

if __name__ == '__main__':
  tf.app.run()
