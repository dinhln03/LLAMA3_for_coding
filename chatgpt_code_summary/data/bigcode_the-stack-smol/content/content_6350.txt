"""
    Training data and validation accuracy.
"""

# Author: Changyu Liu <Shiyipaisizuo@gmail.com>
# Last modified: 2018-07-06
# LICENSE: MIT

import os
import numpy as np
import tensorflow as tf
from PIL import Image

import train_test_split
import cnn


N_CLASSES = 2  # dogs and cats
IMG_W = 208  # resize the image, if the input image is too large, training will be very slow
IMG_H = 208
BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 15000
# with current parameters, it is suggested to use learning rate<0.0001
learning_rate = 0.0001


def run_training():
    # Set there directories .
    train_dir = './data/train/'
    logs_train_dir = './logs/train/'

    train, train_label = train_test_split.get_files(train_dir)

    train_batch, train_label_batch = train_test_split.get_batch(train,
                                                                train_label,
                                                                IMG_W,
                                                                IMG_H,
                                                                BATCH_SIZE,
                                                                CAPACITY)
    train_logits = cnn.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = cnn.losses(train_logits, train_label_batch)
    train_op = cnn.training(train_loss, learning_rate)
    train__acc = cnn.evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])

            if step % 50 == 0:
                print(
                    "Step {}, ".format(step),
                    "train loss = {:.2f}, ".format(tra_loss),
                    "train accuracy = {:.2f}%".format(tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print("Done training -- epoch limit reached")
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


def get_image(train):
    """
        Randomly pick one image from training data
    ====================
    Args:
        train: train data
    ====================
    Return:
        image
    """
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]

    image = Image.open(img_dir)
    image = image.resize([208, 208])
    image = np.array(image)
    return image


def evaluate():
    """
        Test one image against the saved models and parameters
    """

    # you need to change the directories to yours.
    train_dir = './data/train/'
    train, train_label = train_test_split.get_files(train_dir)
    image_array = get_image(train)

    with tf.Graph().as_default():
        batch_size = 1
        n_classes = 2

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 208, 208, 3])
        logits = cnn.inference(image, batch_size, n_classes)

        logits = tf.nn.softmax(logits)

        X = tf.placeholder(tf.float32, shape=[208, 208, 3])

        # you need to change the directories to yours.
        logs_train_dir = './logs/train/'

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split(
                    '/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Loading success, global_step is %s".format(global_step))
            else:
                print("No checkpoint file found")

            prediction = sess.run(logits, feed_dict={X: image_array})
            max_index = np.argmax(prediction)
            if max_index == 0:
                print("This is a cat with possibility {:.6f}".format(
                    prediction[:, 0]))
            else:
                print("This is a dog with possibility {:.6f}".format(
                    prediction[:, 1]))
