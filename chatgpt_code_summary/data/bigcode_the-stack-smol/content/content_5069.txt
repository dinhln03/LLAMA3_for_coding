from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from matplotlib import pyplot as plt
import tensorflow as tf
import seaborn as sb
import pandas as pd
import numpy as np
import math
import time
import cv2
import os


tf.reset_default_graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# tip: if you run into problems with TensorBoard
# clear the contents of this directory, re-run this script
# then restart TensorBoard to see the result
# LOGDIR = './graphs' 
model_frames = 64

NUM_CLASSES = 74
NUM_PIXELS = 88 * 128

TRAIN_STEPS = 0
BATCH_SIZE = 1 << 5

MODEL_ANGLE_DICT = {'000': True, '018': False, '036': False, '054': False, '072': False, '090': False, '108': False, '126': False, '144': False, '162': False, '180': False}
TEST_ANGLE_DICT = {'000': False, '018': False, '036': False, '054': True, '072': False, '090': False, '108': False, '126': False, '144': False, '162': False, '180': False}

LEARNING_RATE = 1e-4

DATA_PATH = 'Generated_full_data_GEI'
start_time = time.time()

keep_prob = 0.5 #dropout (keep probability)


def del_files(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.startswith("."):
                os.remove(os.path.join(root, name))
                print("Delete File: " + os.path.join(root, name))


def get_label(_index, num_classes):
    # label = np.zeros(shape=[num_classes], dtype='float32')
    # label[int(_index) - 1] = 1
    # return label
    return (int(_index) - 1)
    

def load_images_from_folder(folder, model_angle_dict, test_angle_dict):
    train_frames = []
    train_labels = []
    probe_frames = []
    probe_labels = []

    for i in xrange(11):
        train_frames.append([])

    for i in xrange(11):
        train_labels.append([])

    for i in xrange(11):
        probe_frames.append([])

    for i in xrange(11):
        probe_labels.append([])


    for human_id in os.listdir(os.path.join(folder, 'train')):
        if int(human_id) < 74:
            continue
        
        for angle in os.listdir(os.path.join(folder, 'train', human_id)):
            # if not model_angle_dict[angle]:
            #     continue

            for _type in os.listdir(os.path.join(folder, 'train', human_id, angle)):
                img = cv2.imread(os.path.join(folder, 'train', human_id, angle, _type), 0)
                if img is not None:
                    train_frames[int(angle) // 18].append(img.flatten())
                    train_labels[int(angle) // 18].append(get_label(human_id, 124))
                        
    for human_id in os.listdir(os.path.join(folder, 'test')):
        for angle in os.listdir(os.path.join(folder, 'test', human_id)):
            # if not test_angle_dict[angle]:
            #     continue

            for _type in os.listdir(os.path.join(folder, 'test', human_id, angle)):
                img = cv2.imread(os.path.join(folder, 'test', human_id, angle, _type), 0)
                if img is not None:
                    probe_frames[int(angle) // 18].append(img.flatten())
                    probe_labels[int(angle) // 18].append(get_label(human_id, 124))
    
    return (train_frames, train_labels, probe_frames, probe_labels)


del_files(DATA_PATH)
(train_frames, train_labels, probe_frames, probe_labels) = load_images_from_folder(DATA_PATH, MODEL_ANGLE_DICT, TEST_ANGLE_DICT)

# Define inputs
with tf.name_scope('input'):
    images = tf.placeholder(tf.float32, [None, NUM_PIXELS], name="pixels")
    labels = tf.placeholder(tf.float32, [None, NUM_CLASSES], name="labels")

# dropout_prob = tf.placeholder_with_default(1.0, shape=())
    
# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 128, 88, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    conv1 = tf.contrib.layers.batch_norm(conv1)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=3)
    conv2 = tf.contrib.layers.batch_norm(conv2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc3 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc3 = tf.add(tf.matmul(fc3, weights['wd1']), biases['bd1'])
    fc3 = tf.nn.relu(fc3)
    # Apply Dropout
    # fc1 = tf.nn.dropout(fc1, dropout)
    # fc3 = tf.nn.dropout(fc3, dropout_prob)

    # # Output, class prediction
    fc4 = tf.add(tf.matmul(fc3, weights['fc4']), biases['fc4'])
    return fc3

# Store layers weight & bias
initializer = tf.contrib.layers.xavier_initializer()
weights = {
    # 7x7 conv, 1 input, 18 outputs
    'wc1': tf.Variable(initializer([7, 7, 1, 18])),
    # 5x5 conv, 18 inputs, 45 outputs
    'wc2': tf.Variable(initializer([5, 5, 18, 45])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(initializer([32*22*45, 1024])),
    # # 1024 inputs, 10 outputs (class prediction)
    'fc4': tf.Variable(initializer([1024, NUM_CLASSES]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([18])),
    'bc2': tf.Variable(tf.random_normal([45])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'fc4': tf.Variable(tf.random_normal([NUM_CLASSES]))
}
    
y = conv_net(images, weights, biases, keep_prob)

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

saver.restore(sess, "./full_tri_model/model.ckpt")
print("%d frames model restored."%model_frames)

print('   ', end=',')

for i in xrange(11):
    print('%4d'%(i * 18), end=',')

print_map = np.zeros(shape=(11, 11), dtype=np.float32)
gallery_encoding = []
probe_encoding = []

for a in range(11):
    gallery_encoding.append(sess.run(y, feed_dict={images: train_frames[a]}))

for a in range(11):
    probe_encoding.append(sess.run(y, feed_dict={images: probe_frames[a]}))

for a in range(11):
    print('')
    print('%3d'%(a * 18), end=',')

    for b in range(11):     
        simlarity = np.zeros(shape=[len(probe_encoding[b]), len(gallery_encoding[a])], dtype=np.float32)
        pred_label = np.zeros(shape=[len(probe_encoding[b])], dtype=np.int)
        
        for i in range(len(probe_encoding[b])):
            for j in range(len(gallery_encoding[a])):
                simlarity[i][j] = np.exp(-(((probe_encoding[b][i] - gallery_encoding[a][j])/1024.0)**2).sum())

            # import pdb

            # pdb.set_trace()
        
            tmp_index = simlarity[i].argmax()
            pred_label[i] = train_labels[a][tmp_index]
            # if not (pred_label[i] == probe_labels[i]):
            #     print(str((pred_label[i] == probe_labels[i])) + ' ' + str(pred_label[i]) + ' ' + str(probe_labels[i]))
        
        acc = np.sum(pred_label[:] == probe_labels[b][:])
        # print_map[b][10 - a] = 100.0 * acc/(len(probe_labels[b])*1.0)
        print_map[b][a] = 100.0 * acc/(len(probe_labels[b])*1.0)
        print('%.2f'%(100.0 * acc/(len(probe_labels[b])*1.0)), end=',')
print(print_map)


grid_visualization = np.array(print_map.transpose())
grid_visualization.shape = (11, 11)
sb.heatmap(grid_visualization, cmap='Oranges')
plt.xticks(np.arange(11) + 0.5, xrange(0, 181, 18))
plt.yticks(np.arange(11) + 0.5, xrange(180, -1, -18))
plt.xlabel('Gallery Angle')
plt.ylabel('Probe Angle')

plt.show()