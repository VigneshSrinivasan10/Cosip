import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
#from caffe_classes import class_names
import pdb
import glob
import os
import sys

flags = tf.flags

flags.DEFINE_string("model_path",'vgg16_weights.npz', "")
flags.DEFINE_string("test_img_path",'ILSVRC2012_val_00012237.JPEG', "")

FLAGS = flags.FLAGS


class vgg16:
    def __init__(self, weights=None, sess=None):
        self.weights = weights
        self.sess = sess 
        # self.forward(imgs)
        #pass
    
    def forward(self, imgs):
        
        end_points = self.all_layers(imgs)
        #self.fc_layers()
        #self.probs = tf.nn.softmax(self.fc3l)
        #end_points['prob'] = self.probs
        return end_points#, self.probs
    
    def all_layers(self, imgs):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = imgs-mean

        end_points = {}
        # conv1_1
        with tf.variable_scope('conv1_1') as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(),  shape=[3, 3, 3, 64], name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[64], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]
            end_points['conv1_1'] = self.conv1_1
            
        # conv1_2
        with tf.variable_scope('conv1_2') as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(),  shape=[3, 3, 64, 64], name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[64], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]
            end_points['conv1_2'] = self.conv1_2
        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')
        end_points['pool1'] = self.pool1
        # conv2_1
        with tf.variable_scope('conv2_1') as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(),  shape=[3, 3, 64, 128], name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[128], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]
            end_points['conv2_1'] = self.conv2_1
        # conv2_2
        with tf.variable_scope('conv2_2') as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(),  shape=[3, 3, 128, 128], name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[128], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]
            end_points['conv2_2'] = self.conv2_2
        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')
        end_points['pool2'] = self.pool2
        
        # conv3_1
        with tf.variable_scope('conv3_1') as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(),  shape=[3, 3, 128, 256], name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[256], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]
            end_points['conv3_1'] = self.conv3_1
        # conv3_2
        with tf.variable_scope('conv3_2') as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(),  shape=[3, 3, 256, 256], name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[256], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]
            end_points['conv3_2'] = self.conv3_2
        # conv3_3
        with tf.variable_scope('conv3_3') as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(),  shape=[3, 3, 256, 256], name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[256], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out)
            self.parameters += [kernel, biases]
            end_points['conv3_3'] = self.conv3_3
        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')
        end_points['pool3'] = self.pool3
        
        # conv4_1
        with tf.variable_scope('conv4_1') as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(),  shape=[3, 3, 256, 512], name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[512], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]
            end_points['conv4_1'] = self.conv4_1
        # conv4_2
        with tf.variable_scope('conv4_2') as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(),  shape=[3, 3, 512, 512], name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[512], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]
            end_points['conv4_2'] = self.conv4_2
        # conv4_3
        with tf.variable_scope('conv4_3') as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(),  shape=[3, 3, 512, 512], name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[512], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out)
            self.parameters += [kernel, biases]
            end_points['conv4_3'] = self.conv4_3
        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')
        end_points['pool4'] = self.pool4
        
        # conv5_1
        with tf.variable_scope('conv5_1') as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(),  shape=[3, 3, 512, 512], name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[512], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]
            end_points['conv5_1'] = self.conv5_1
        # conv5_2
        with tf.variable_scope('conv5_2') as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(),  shape=[3, 3, 512, 512], name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[512], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]
            end_points['conv5_2'] = self.conv5_2
        # conv5_3
        with tf.variable_scope('conv5_3') as scope:
            kernel = tf.get_variable(initializer=tf.truncated_normal_initializer(),  shape=[3, 3, 512, 512], name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[512], dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out)
            self.parameters += [kernel, biases]
            end_points['conv5_3'] = self.conv5_3
        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')
        end_points['pool5'] = self.pool5
        '''
        #def fc_layers(self):
        # fc1
        with tf.variable_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.get_variable(initializer=tf.truncated_normal_initializer(),  shape=[shape, 4096],
                                                          name='weights')
            fc1b = tf.get_variable(initializer=tf.constant_initializer(1.0), shape=[4096], dtype=tf.float32, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]
            end_points['fc6'] = self.fc1
        # fc2
        with tf.variable_scope('fc2') as scope:
            fc2w = tf.get_variable(initializer=tf.truncated_normal_initializer(),  shape=[4096, 4096],
                                                          name='weights')
            fc2b = tf.get_variable(initializer=tf.constant_initializer(1.0), shape=[4096], dtype=tf.float32, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]
            end_points['fc7'] = self.fc2
        # fc3
        with tf.variable_scope('fc3') as scope:
            fc3w = tf.get_variable(initializer=tf.truncated_normal_initializer(),  shape=[4096, 1000],
                                                          name='weights')
            fc3b = tf.get_variable(initializer=tf.constant_initializer(1.0), shape=[1000], dtype=tf.float32, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]
            end_points['fc8'] = self.fc3l
	'''
        return end_points
    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        print('Reloading VGG16 weights ...')
        for i, k in enumerate(keys):
            if 'fc' not in k:
                sess.run(self.parameters[i].assign(weights[k]))


def batchGen(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def main():
    
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    with tf.variable_scope('vgg') as scope:
        model = vgg16()
        end_points, vgg_pred = model.forward(imgs)
    model.load_weights(FLAGS.model_path,sess)

    img_file = FLAGS.test_img_path
    img1 = imread(img_file)[:,:,:3]
    
    img1 = imresize(img1, (224, 224))
    prob = sess.run(vgg_pred, feed_dict={imgs: [img1]})[0]

    class_name = class_names[np.where(prob==prob.max())[0][0]] 
    print('Predicted class: %s with a probability of: ' % class_name, prob.max())
    
if __name__=="__main__":
    main()
