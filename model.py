import os
import time
from glob import glob
import tensorflow as tf

#from tensorflow.python.ops.script_ops import *
from ops import *
from utils import *
from compute_ei import *

class DCGAN(object):
    def __init__(self, sess, image_size=108, is_crop=True,
                 batch_size=32, input_size=64, sample_size=32, ir_image_shape=[64, 64,1], normal_image_shape=[64, 64, 3],
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

        self.lambda1 = 1
        self.lambda2 = 1

        self.c_dim = 3

        if input_size != ir_image_shape[0]:
            ir_image_shape[0] = input_size
            ir_image_shape[1] = input_size
            normal_image_shape[0] = input_size
            normal_image_shape[1] = input_size

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(batch_size, name='d_bn1')
        self.d_bn2 = batch_norm(batch_size, name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(batch_size, name='d_bn3')

        self.g_bn0 = batch_norm(batch_size, name='g_bn0')
        self.g_bn1 = batch_norm(batch_size, name='g_bn1')
        self.g_bn2 = batch_norm(batch_size, name='g_bn2')
        self.g_bn3 = batch_norm(batch_size, name='g_bn3')
        self.g_bn4 = batch_norm(batch_size, name='g_bn4')

        if not self.y_dim:
            self.g_bn3 = batch_norm(batch_size, name='g_bn3')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y= tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        self.ir_images = tf.placeholder(tf.float32, [self.batch_size] + self.ir_image_shape,
                                    name='ir_images')
        self.normal_images = tf.placeholder(tf.float32, [self.batch_size] + self.normal_image_shape,
                                    name='normal_images')
        self.ir_sample_images= tf.placeholder(tf.float32, [self.sample_size] + self.ir_image_shape,
                                        name='ir_sample_images')
        self.ei_images = tf.placeholder(tf.float32, [self.batch_size] + self.ir_image_shape,
                                    name='ei_images')


        self.G = self.generator(self.ir_images)
        self.D = self.discriminator(self.normal_images) # real image output
        self.sampler = self.sampler(self.ir_images)
        self.D_ = self.discriminator(self.G, reuse=True) #fake image output
        self.d_sum = tf.histogram_summary("d", self.D)
        self.d__sum = tf.histogram_summary("d_", self.D_)
        #self.G_sum = tf.image_summary("G", self.G)

        self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.D), self.D)
        self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_), self.D_)
        self.L1_loss = tf.reduce_sum(tf.abs(tf.sub(self.G,self.normal_images)))/self.batch_size * self.lambda2
        self.EI_loss = tf.py_func(compute_ei,[self.G],[tf.float64])
        self.EI_loss = tf.to_float(self.EI_loss[0],name='ToFloat')
        self.g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.D_), self.D_)*self.lambda1 + self.L1_loss + tf.abs(self.EI_loss)

        self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        """Train DCGAN"""

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        tf.initialize_all_variables().run()

        self.saver = tf.train.Saver()
        self.g_sum = tf.merge_summary([self.d__sum,
        self.d_loss_fake_sum, self.g_loss_sum])
        self.g_sum = tf.merge_summary([self.d__sum,self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.merge_summary([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph_def)

        # Evaluation
        self.val_g_sum = tf.merge_summary([self.d__sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.val_writer = tf.train.SummaryWriter("./val_logs", self.sess.graph_def)

        val_data = json.load(open("/home/yjyoon/work/PycharmProjects/ECCV2016/DCGAN_IR_single/testinput_single.json"))
        val_label = json.load(open("/home/yjyoon/work/PycharmProjects/ECCV2016/DCGAN_IR_single/testgt_single.json"))
        val_datalist =[', '.join(val_data[idx]) for idx in xrange(0,len(val_data))]
        val_labellist =[', '.join(val_label[idx]) for idx in xrange(0,len(val_data))]

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # loda training and validation dataset path
        data = json.load(open("/home/yjyoon/work/PycharmProjects/ECCV2016/DCGAN_IR_single/traininput_single.json"))
        data_label = json.load(open("/home/yjyoon/work/PycharmProjects/ECCV2016/DCGAN_IR_single/traingt_single.json"))
        datalist =[', '.join(data[idx]) for idx in xrange(0,len(data))]
        labellist =[', '.join(data_label[idx]) for idx in xrange(0,len(data))]

	#print('Load all images')
	#load_all_images(data,data_label,self.batch_size,config.train_size)	


        for epoch in xrange(config.epoch):
            # loda training and validation dataset path
            shuffle = np.random.permutation(range(len(data)))
            batch_idxs = min(len(data), config.train_size)/config.batch_size
            randx = np.random.randint(64,224-64)
            randy = np.random.randint(64,224-64)
            for idx in xrange(0, batch_idxs):
                batch_files = shuffle[idx*config.batch_size:(idx+1)*config.batch_size]
                ir_batch = [get_image(datalist[batch_file], self.image_size,randx,randy, is_crop=self.is_crop,gray=True)
                            for batch_file in batch_files]

                normal_batchlabel = [get_image(labellist[batch_file], self.image_size,randx,randy,
                                                      is_crop=self.is_crop,gray=False) for batch_file in batch_files]

                batch_images = np.array(ir_batch).astype(np.float32)
                batchlabel_images = np.array(normal_batchlabel).astype(np.float32)


                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={self.normal_images: batchlabel_images,
                                                                                 self.ir_images: batch_images })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={ self.ir_images: batch_images,self.normal_images: batchlabel_images })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={ self.ir_images: batch_images,self.normal_images: batchlabel_images  })
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.ir_images: batch_images})
                errD_real = self.d_loss_real.eval({self.normal_images: batchlabel_images})
                errG = self.g_loss.eval({self.ir_images: batch_images, self.normal_images: batchlabel_images})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))
                if np.mod(counter, batch_idxs) == 1:
                    valrandx = np.random.randint(64,224-64)
                    valrandy = np.random.randint(64,224-64)
                    valshuffle = np.random.permutation(range(len(val_data)))

                    batch_files = valshuffle[0:self.sample_size]
                    ir_batch = [get_image(val_datalist[batch_file], self.image_size,valrandx,valrandy, is_crop=self.is_crop) for batch_file in batch_files]
                    normal_batchlabel = [get_image(val_labellist[batch_file], self.image_size,valrandx,valrandy, is_crop=self.is_crop) \
                             for batch_file in batch_files]

                    val_batch_images = np.array(ir_batch).astype(np.float32)
                    val_batchlabel_images = np.array(normal_batchlabel).astype(np.float32)

                    samples, summary_str = self.sess.run([self.sampler, self.val_g_sum], feed_dict={self.ir_images: val_batch_images,
                                                                                 self.normal_images: val_batchlabel_images})
                    self.val_writer.add_summary(summary_str, counter)
                    errG = self.g_loss.eval({self.ir_images: val_batch_images, self.normal_images: val_batchlabel_images})
                    save_images(samples, [8, 8],'./samples/train_%s_%s.png' % ( epoch, idx))
                    save_images(val_batchlabel_images, [8, 8],'./samples/gt_%s_%s.png' % (epoch, idx))
                    print("[Sample] g_loss: %.8f" % ( errG))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def discriminator(self, image, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        if not self.y_dim:
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4)
        else:
            yb = tf.reshape(y, [None, 1, 1, self.y_dim])
            x = conv_cond_concat(image, yb)

            h0 = lrelu(spatial_conv(x, self.c_dim + self.y_dim))
            h0 = conv_cond_concat(h0, yb)

            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim)))
            h1 = tf.reshape(h1, [h1.get_shape()[0], -1])
            h1 = tf.concat(1, [h1, y])

            h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
            h2 = tf.concat(1, [h2, y])

            return tf.nn.sigmoid(linear(h2, 1, 'd_h3_lin'))

    def generator(self, real_image, y=None):
        if not self.y_dim:

            h1 = conv2d(real_image,self.gf_dim*2,d_h=1,d_w=1, name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1))

            h2 = conv2d(h1,self.gf_dim*4, d_h=1,d_w=1, name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3 = conv2d(h2,self.gf_dim*4, d_h=1,d_w=1, name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4 = conv2d(h3, self.gf_dim*2, d_h=1,d_w=1, name='g_h4')
            h4 = tf.nn.relu(self.g_bn4(h4))

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
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))

            h2 = conv2d(h1, self.gf_dim*4,d_h=1,d_w=1, name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))

            h3 = conv2d(h2,self.gf_dim*4, d_h=1,d_w=1, name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4 = conv2d(h3, self.gf_dim*2,d_h=1,d_w =1, name='g_h4')
            h4 = tf.nn.relu(self.g_bn4(h4, train=False))

            h5 = conv2d(h4, 3,d_h=1,d_w=1, name='g_h5')

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

        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

             
