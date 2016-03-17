import os
import time
from glob import glob
import tensorflow as tf

#from tensorflow.python.ops.script_ops import *
from ops import *
from utils import *
from compute_ei import *

class EVAL(object):
    def __init__(self, sess, image_size=108, is_crop=True,
                 batch_size=1, input_size=64, sample_size=32, ir_image_shape=[64, 64,1], normal_image_shape=[64, 64, 3],
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 checkpoint_dir=None):

        """

        Args:
            input_size (object):
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.normal_image_shape = normal_image_shape
        self.ir_image_shape = ir_image_shape

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.lambda_g = 1.
        self.lambda_l2 = 10.
        self.lambda_ei = 0.1
        self.c_dim = 3
        """
        if input_size != ir_image_shape[0]:
            ir_image_shape[0] = input_size
            ir_image_shape[1] = input_size
            normal_image_shape[0] = input_size
            normal_image_shape[1] = input_size
        """

        self.g_bn0 = batch_norm(batch_size, name='g_bn0')
        self.g_bn1 = batch_norm(batch_size, name='g_bn1')
        self.g_bn2 = batch_norm(batch_size, name='g_bn2')
        self.g_bn3 = batch_norm(batch_size, name='g_bn3')
        self.g_bn4 = batch_norm(batch_size, name='g_bn4')
        self.g_bn5 = batch_norm(batch_size, name='g_bn5')
        self.g_bn6 = batch_norm(batch_size, name='g_bn6')

        if not self.y_dim:
            self.g_bn3 = batch_norm(batch_size, name='g_bn3')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):

        self.ir_images = tf.placeholder(tf.float32, [self.batch_size] + self.ir_image_shape,
                                    name='ir_images')
        self.normal_images = tf.placeholder(tf.float32, [self.batch_size] + self.normal_image_shape,
                                    name='normal_images')

        self.G = self.generator(self.ir_images)
        self.sampler = self.sampler(self.ir_images)

        self.saver = tf.train.Saver()

    def generator(self, real_image, y=None):
        if not self.y_dim:

            h1 = conv2d(real_image,self.gf_dim*2,d_h=1,d_w=1, name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1))
            
            h2 = conv2d(h1,self.gf_dim*4,d_h=1,d_w=1, name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2))
           
            h3 = conv2d(h2,self.gf_dim*4,d_h=1,d_w=1, name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3))
            
            h4 = conv2d(h3,self.gf_dim*2,d_h=1,d_w=1, name='g_h4')
            h4 = tf.nn.relu(self.g_bn4(h4))
            """
            h2 = bottleneck_block_letters('2',h1,1,self.gf_dim*2,self.gf_dim*4)
            
            h3 = bottleneck_block_letters('3',h2,1,self.gf_dim*4,self.gf_dim*8)
            
            h4 = bottleneck_block_letters('4',h3,1,self.gf_dim*8,self.gf_dim*4)
            """
            h5 = conv2d(h4,3, d_h=1,d_w=1, name='g_h5')
   
            return tf.nn.tanh(h5)
        else:
            yb = tf.reshape(y, [None, 1, 1, self.y_dim])
            z = tf.concat(1, [z, y])

            h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
            h0 = tf.concat(1, [h0, y])

            h1 = tf.nn.relu(self.g_bn1(linear(z, self.gf_dim*2*7*7, 'g_h1_lin')))
            h1 = tf.reshape(h1, [None, 7, 7, self.gf_dim * 2])
            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, self.gf_dim, name='g_h2')))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, self.c_dim, name='g_h3'))



    def sampler(self,images, y=None):
        tf.get_variable_scope().reuse_variables()

        if not self.y_dim:
            
            h1 = conv2d(images,self.gf_dim*2,d_h=1,d_w=1, name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1,train=False))
            
            h2 = conv2d(h1,self.gf_dim*4,d_h=1,d_w=1, name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2,train=False))
           
            h3 = conv2d(h2,self.gf_dim*4,d_h=1,d_w=1, name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3,train=False))
            
            h4 = conv2d(h3,self.gf_dim*2,d_h=1,d_w=1, name='g_h4')
            h4 = tf.nn.relu(self.g_bn4(h4,train=False))


            h5 = conv2d(h4,3, d_h=1,d_w=1, name='g_h5')
            return tf.nn.tanh(h5)
        else:
            yb = tf.reshape(y, [None, 1, 1, self.y_dim])
            z = tf.concat(1, [z, y])

            h0 = tf.nn.relu(self.bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
            h0 = tf.concat(1, [h0, y])

            h1 = tf.nn.relu(self.g_bn1(linear(z, self.gf_dim*2*7*7, 'g_h1_lin')))
            h1 = tf.reshape(h1, [None, 7, 7, self.gf_dim * 2])
            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(self.bn2(deconv2d(h1, self.gf_dim, name='g_h2')))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, self.c_dim, name='g_h3'))

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        #model_dir = "%s_%s" % (self.dataset_name, 32)
        model_dir = "%s" % (self.dataset_name)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print('*************** ckpt *************')
        print(ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.all_model_checkpoint_paths[-3])
            print('Loaded network:',ckpt_name)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

             
