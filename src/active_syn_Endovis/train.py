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
tf.set_random_seed(23)

flags = tf.app.flags
flags.DEFINE_integer("epoch",20, "Epoch to train [25]")
flags.DEFINE_integer("batch_size", 16, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 240, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", 240, "The size of image to use (will be center cropped). If None, same value as input_height [None]")

# the folder of training set and validation set, must contain a sub-folder called 'images' and a sub-folder called 'labels' 
flags.DEFINE_string("train_dataset", "D:/uw/PHD/2020summer/instrument segmentation/code_dev/synthetic_endo_img_generator_v3/data_frame_12_29_5", "train dataset direction")
flags.DEFINE_string("val_dataset", "D:/mafa_deeplab_Shan/code/uw-sinus-surgery-CL/cadaver", "train dataset direction")

# flags.DEFINE_string("train_dataset", "D:/uw/PHD/2020summer/instrument segmentation/code_dev/synthetic_endo_img_generator_v2/data_frame_11_23", "train dataset direction")
# flags.DEFINE_string("val_dataset", "D:/mafa_deeplab_Shan/code/uw-sinus-surgery-CL/cadaver", "train dataset direction")

# flags.DEFINE_string("train_dataset", "D:/mafa_deeplab_Shan/code/uw-sinus-surgery-CL/cadaver_train", "train dataset direction")
# flags.DEFINE_string("val_dataset", "D:/mafa_deeplab_Shan/code/uw-sinus-surgery-CL/cadaver_test", "train dataset direction")

flags.DEFINE_string("img_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("label_pattern", "*.png", "Glob pattern of filename of input labels [*]")
flags.DEFINE_string("checkpoint_dir", "./checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("pretrain_dir", "./mobilenet_v1_1.0_224", "")
flags.DEFINE_string("gpu", '0', "gpu")
FLAGS = flags.FLAGS


os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
def main(_):
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

      
      

    
if __name__ == '__main__':
  tf.app.run()
