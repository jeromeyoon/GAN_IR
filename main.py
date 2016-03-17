import os
import random
import numpy as np
import tensorflow as tf
import time 
import json
from model import DCGAN
from test import EVAL
from utils import pp, save_images, to_json, make_gif, merge, imread, get_image
import scipy.misc
from numpy import inf
flags = tf.app.flags
flags.DEFINE_integer("epoch", 1000, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_integer("input_size", 64, "The size of image input size")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if FLAGS.is_train:
            dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size, input_size=FLAGS.input_size,
                      dataset_name=FLAGS.dataset,
                      is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir)
        else:
   
            dcgan = EVAL(sess, input_size = 600, batch_size=1,ir_image_shape=[600,800,1],normal_image_shape=[600,800,3],dataset_name=FLAGS.dataset,\
                      is_crop=False, checkpoint_dir=FLAGS.checkpoint_dir)

        if FLAGS.is_train:
            dcgan.train(FLAGS)
        else:
            dcgan.load(FLAGS.checkpoint_dir)
            OPTION = 2 # for validation
            list_val = [11,16,21,22,33,36,38,53,59,92]
            VAL_OPTION =2
            """
            if OPTION == 1:
                data = json.load(open("/research2/IR_normal_small/json/traininput_single_224_ori_small.json"))
                data_label = json.load(open("/research2/IR_normal_small/json/traingt_single_224_ori_small.json"))
            
            elif OPTION == 2:
                data = json.load(open("/research2/IR_normal_small/json/testinput_single_224_ori_small.json"))
                data_label = json.load(open("/research2/IR_normal_small/json/testgt_single_224_ori_small.json"))
            """
            if VAL_OPTION ==1:
                list_val = [11,16,21,22,33,36,38,53,59,92]
                for idx in range(len(list_val)):
                    for idx2 in range(1,10): 
                        print("Selected material %03d/%d" % (list_val[idx],idx2))
                        img = '/research2/IR_normal_small/save%03d/%d' % (list_val[idx],idx2)
                        input_ = scipy.misc.imread(img+'/3.bmp').astype(float)
                        gt_ = scipy.misc.imread('/research2/IR_normal_small/save016/1/12_Normal.bmp').astype(float)
                        input_ = scipy.misc.imresize(input_,[600,800])
                        gt_ = scipy.misc.imresize(gt_,[600,800])
                        #input_ = input_[240:840,515:1315]
                        #gt_ = gt_[240:840,515:1315]
                        input_ = np.reshape(input_,(1,600,800,1)) 
                        gt_ = np.reshape(gt_,(1,600,800,3)) 
                        input_ = np.array(input_).astype(np.float32)
                        gt_ = np.array(gt_).astype(np.float32)
                        start_time = time.time() 
                        sample = sess.run(dcgan.sampler, feed_dict={dcgan.ir_images: input_})
                        print('time: %.8f' %(time.time()-start_time))     
                        # normalization #
                        sample = np.squeeze(sample).astype(np.float32)
                        gt_ = np.squeeze(gt_).astype(np.float32)

                        output = np.zeros((600,800,3)).astype(np.float32)
                        output[:,:,0] = sample[:,:,0]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
                        output[:,:,1] = sample[:,:,1]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
                        output[:,:,2] = sample[:,:,2]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
   
                        output[output ==inf] = 0.0
                        sample = (output+1.)/2.
                        savename = '/home/yjyoon/Dropbox/ECCV16_IRNormal/single_result/%03d/%d/single_normal_L2ang.bmp' % (list_val[idx],idx2)

                        scipy.misc.imsave(savename, sample)

            
            elif VAL_OPTION ==2:
                print("Computing all validation set ")
                ErrG =0.0
		num_img =13
                for idx in xrange(5, num_img+1):
                    print("[Computing Validation Error %d/%d]" % (idx, num_img))
                    img = '/home/yjyoon/Dropbox/ECCV16_IRNormal/extra/extra_%d.bmp' % (idx)
                    input_ = scipy.misc.imread(img).astype(float)
                    input_ = input_[:,:,0]
                    gt_ = scipy.misc.imread('/research2/IR_normal_small/save016/1/12_Normal.bmp').astype(float)
                    input_ = scipy.misc.imresize(input_,[600,800])
                    gt_ = scipy.misc.imresize(gt_,[600,800])
                    input_ = np.reshape(input_,(1,600,800,1)) 
                    gt_ = np.reshape(gt_,(1,600,800,3)) 
                    input_ = np.array(input_).astype(np.float32)
                    gt_ = np.array(gt_).astype(np.float32)
                    start_time = time.time() 
                    sample = sess.run(dcgan.sampler, feed_dict={dcgan.ir_images: input_})
                    print('time: %.8f' %(time.time()-start_time))     
                    # normalization #
                    sample = np.squeeze(sample).astype(np.float32)
                    gt_ = np.squeeze(gt_).astype(np.float32)

                    output = np.zeros((600,800,3)).astype(np.float32)
                    output[:,:,0] = sample[:,:,0]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
                    output[:,:,1] = sample[:,:,1]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
                    output[:,:,2] = sample[:,:,2]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
   
                    output[output ==inf] = 0.0
                    sample = (output+1.)/2.
                    savename = '/home/yjyoon/Dropbox/ECCV16_IRNormal/extra/extra_result%d.bmp' % (idx)

                    scipy.misc.imsave(savename, sample)


if __name__ == '__main__':
    tf.app.run()
