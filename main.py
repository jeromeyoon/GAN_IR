import os
import random
import numpy as np
import tensorflow as tf
from time import gmtime, strftime
import json
from model import DCGAN
from utils import pp, save_images, to_json, make_gif, merge, imread, get_image
import scipy.misc

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1000, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.002, "Learning rate of for adam [0.0002]")
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
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size, input_size=FLAGS.input_size,
                      dataset_name=FLAGS.dataset,
                      is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir)

        if FLAGS.is_train:
            dcgan.train(FLAGS)
        else:
            dcgan.load(FLAGS.checkpoint_dir)
            OPTION = 2
            if OPTION == 1:
                data = json.load(open("/home/yjyoon/work/IRnormal/data/traininput_material.json"))
                data_label = json.load(open("/home/yjyoon/work/IRnormal/data/traingt_material.json"))
                """
                data= glob.glob('research1/db/IR_normal_small/save010/1')
                im1 = scipy.misc.imread(data[0])
                im2 = scipy.misc.imread(data[1])
                im3 = scipy.misc.imread(data[2])
                im = np.dstack(im1,im2,im3)
                """
            elif OPTION == 2:
                data = json.load(open("/home/yjyoon/work/IRnormal/data/testinput_material.json"))
                data_label = json.load(open("/home/yjyoon/work/IRnormal/data/testgt_material.json"))

            shuffle = np.random.permutation(range(len(data)))
            ir_batch = [get_image(data[shuffle[idx]], 0, 64, 64, is_crop=FLAGS.is_crop) for idx in xrange(FLAGS.batch_size)]
            normal_batchlabel = [get_image(data_label[shuffle[idx]], 0, 64, 64, is_crop=FLAGS.is_crop) for idx in xrange(FLAGS.batch_size)]
            eval_batch_images = np.array(ir_batch).astype(np.float32)
            eval_batchlabel_images = np.array(normal_batchlabel).astype(np.float32)
            samples = sess.run(dcgan.sampler, feed_dict={dcgan.ir_images: eval_batch_images})
            h = eval_batch_images.shape[1]
            w = eval_batch_images.shape[2]
            img = np.zeros((h, w * 2, 3))
            for idx in xrange(FLAGS.batch_size):
                predict = samples[idx, :, :, :]
                gt = eval_batchlabel_images[idx, :, :, :]
                print('error:', sess.run(tf.reduce_sum(tf.abs(tf.sub(predict, gt)))))
                img[0:h, 0:w, :] = predict
                img[0:h, w:2 * w, :] = gt
                scipy.misc.imshow(img)


if __name__ == '__main__':
    tf.app.run()
