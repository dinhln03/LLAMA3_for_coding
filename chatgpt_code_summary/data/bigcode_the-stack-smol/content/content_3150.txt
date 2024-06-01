from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import os.path
import math

import tensorflow as tf
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

LOGDIR = "/tmp/cnn_backbone_angles/"

# Parameters
batch_size = 5
training_epochs = 10
display_step = 1
internal_channels_1 = 100
internal_channels_2 = 100
internal_channels_3 = 100
internal_channels_4 = 50

window_size = 11
beta = 0.001
values_to_predict = 2
num_splits = 10
alpha = 0.2
dropout_keep_rate = 0.5
learning_rate = 1E-3
keep_prob = tf.placeholder_with_default(1.0, shape=(), name="keep_prob")
keep_prob_input = tf.placeholder_with_default(1.0, shape=(), name="keep_prob_input")

def fc_layer(input, size_in, size_out, name="fc"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([window_size, size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    act = conv1d(input, w) + b
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return act, w


def convnn(x, channels_num, layers_num, window_size = 11):
    W_arr = []
    layers = []
    # First convolutional layer
    input_dimensions = x.get_shape().as_list()[1:]
    filter_shape = [window_size, input_dimensions[-1], channels_num]
    W_input = weight_variable(filter_shape)
    W_arr.append(W_input)
    b_input = bias_variable([input_dimensions[0], channels_num])
    input_layer = tf.nn.relu(conv1d(x, W_input) + b_input)
    dropout_input = tf.nn.dropout(input_layer, keep_prob_input)
    layers.append(dropout_input)
    # Hidden layers
    filter_shape = [window_size, channels_num, channels_num]
    W_hidden = tf.constant([], dtype=tf.float32)
    for i in range(layers_num):
        with tf.name_scope("conv"):
            W_hidden = weight_variable(filter_shape)
            W_arr.append(W_hidden)
            b_hidden = bias_variable([input_dimensions[0], channels_num])
            conv_layer = tf.nn.tanh(alpha*conv1d(layers[i], W_hidden) + b_hidden)
            tf.summary.histogram("weights", W_hidden)
            tf.summary.histogram("biases", b_hidden)
            tf.summary.histogram("activations", conv_layer)
        with tf.name_scope("dropout"):
            dropout = tf.nn.dropout(conv_layer, keep_prob)
            layers.append(dropout)
    # Output convolutional layer
    layer_out, W_out = fc_layer(layers[-1], channels_num, values_to_predict)
    W_arr.append(W_out)
    # layer_out =  tf.atan2(tf.sin(layer_out), tf.cos(layer_out))

    # Loss function with L2 Regularization with beta=0.001
    regularizers = tf.nn.l2_loss(W_input) + tf.nn.l2_loss(W_hidden) * layers_num + tf.nn.l2_loss(W_out)

    # regularizers = tf.constant(0, dtype=tf.float32)
    # for W in W_arr:
    #     regularizers += tf.nn.l2_loss(W)

    return layer_out, regularizers

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name="W")


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name="B")

def conv1d(x, W):
    """conv1d returns a 1d convolution layer."""
    return tf.nn.conv1d(x, W, 1, 'SAME')

def avgpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def calculate_accuracy(predictions, labels):
    num_proteins = predictions.shape[0]
    protein_accuracy = np.zeros(num_proteins, dtype=np.float32)
    label_accuracy = {1: {"total": 0, "correct": 0}, 2: {"total": 0, "correct": 0},
                      3: {"total": 0, "correct": 0}}
    for i in range(num_proteins):
        total_predictions = 0
        correct_predictions = 0
        for j in range(predictions.shape[1]):
            phi = math.degrees(labels[i][j][0])
            phi0 = math.degrees(predictions[i][j][0])
            psi = math.degrees(labels[i][j][1])
            psi0 = math.degrees(predictions[i][j][1])
            if (phi != 0) or (psi != 0):
                total_predictions += 1
                expected_state = get_backbone_distribution(labels[i][j])
                predicted_state = get_backbone_distribution(predictions[i][j])
                label_accuracy[predicted_state]["total"] += 1
                if (predicted_state == expected_state):
                    # correct_predictions += 1
                    label_accuracy[predicted_state]["correct"] += 1
                # print("REAL PHI->>>>>"+str(labels[i][j][0]))
                # print("PREDICTED PHI->>>>>" + str(predictions[i][j][0]))
                diff = math.sqrt(math.pow(phi - phi0, 2)+math.pow(psi - psi0, 2))
                diff_phi = phi0 - phi0
                diff_psi = psi - psi0
                criteria_1 = (np.abs(diff_phi) < 60) & (np.abs(diff_psi) < 60)
                criteria_2 = (np.abs(diff_phi+diff_psi) < 60) & (np.abs(diff_psi) < 90) & (np.abs(diff_phi) < 90)
                if (diff < 60):
                    correct_predictions += 1
            # print("CORRECT->>>>>"+str(correct_predictions))
            # print("TOTAL->>>>>" + str(total_predictions))
            if (total_predictions > 0):
                protein_accuracy[i] = correct_predictions / float(total_predictions)

    accuracy_dist = {}
    total = 0
    correct = 0
    for label, val in label_accuracy.iteritems():
        if (val["total"] > 0):
            accuracy_dist[label] = val["correct"]/val["total"]
            total += val["total"]
            correct += val["correct"]
    if (total > 0):
        accuracy_dist["total"] = correct/total
    return protein_accuracy, accuracy_dist

def get_backbone_distribution(angles):
    phi = math.degrees(angles[0])
    psi = math.degrees(angles[1])
    #  A: -160 < phi <0 and -70 < psi < 60
    if (-160 < phi < 0) & (-70 < psi < 60):
        return 1
    # P: 0 < phi < 160 and -60 < psi < 95
    elif (0 < phi < 160) & (-60 < psi < 95):
        return 2
    else:
        return 3

def plot_ramachandran(predictions, title):
    phi_angles = predictions[:][:][0].flatten()
    phi_angles = list(map(lambda x: math.degrees(x), phi_angles))
    psi_angles = predictions[:][:][1].flatten()
    psi_angles = list(map(lambda x: math.degrees(x), psi_angles))
    colors = np.random.rand(len(psi_angles))
    fig = plt.figure()
    plt.xlim([-180, 180])
    plt.ylim([-180, 180])
    plt.title(title)
    plt.xlabel('phi')
    plt.ylabel('psi')
    plt.grid()
    plt.scatter(phi_angles, psi_angles, alpha=0.5, c=colors)
    fig.savefig("./plots/" + title + ".png", bbox_inches='tight')
    # plt.show()
    # fig.savefig("./plots/" + title + ".png", bbox_inches='tight')
    plt.close()

def plot_loss(loss_arr):
    l = plt.figure()
    plt.plot(loss_arr)
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(plot_legend, loc='upper left')
    l.show()

def make_hparam_string(layers_num, channels_num, test_session):
  return "nl_%s,nc_%s, session%s" % (layers_num, channels_num, test_session)

def convert_to_degrees(arr):
    """Covert all phi and psi angles to degrees"""
    arr[0] = math.degrees(arr[0])
    arr[1] = math.degrees(arr[1])
    return arr

data = np.load('phipsi_features.npz')['features']
all_data = data.reshape(data.shape[0],700,69)
# all_data = all_data[0:300]
all_sets = all_data[:,:,0:21]
all_sets = np.concatenate([all_sets, all_data[:,:,21:42]], axis=-1)
all_sets = np.concatenate([all_sets, all_data[:,:,42:63]], axis=-1)
# all_labels = all_data[:,:,63:67]
all_angles = all_data[:,:,67:69]
where_are_NaNs = np.isnan(all_angles)
all_angles[where_are_NaNs] = 0.0
k_fold = KFold(n_splits=num_splits)

layers_channels = [(6, 100), (7, 100)]
# Build the convolutional network
for layers_num, channels_num in layers_channels:
    for use_l2 in [False, True]:
        for use_early_stopping in [True, False]:
            crossvalidation_train_accuracy = 0
            crossvalidation_test_accuracy = 0
            crossvalidation_accuracy_distr = {'total': 0, 1: 0, 2: 0, 3: 0}
            crossvalidation_test_mae = 0
            executed_epochs = 0
            train_session = 0
            test_session = 0
            learning_rate_type = 1
            for train_index, test_index in k_fold.split(all_sets):
                train_set, test_set = all_sets[train_index], all_sets[test_index]
                train_labels, test_labels = all_angles[train_index], all_angles[test_index]
                train_size = train_set.shape[0]
                train_y = train_labels
                test_y = test_labels
                test_session += 1

                # Create the model
                x = tf.placeholder(tf.float32, [None, 700, train_set[0].shape[-1]], name="x")

                # Define loss and optimizer
                y_ = tf.placeholder(tf.float32, [None, 700, values_to_predict], name="labels")

                y_nn, regularizers = convnn(x, channels_num, layers_num, window_size)
                prediction = y_nn

                with tf.name_scope("loss"):
                    deviations = tf.subtract(prediction, y_)
                    ae = tf.abs(deviations)
                    mae = tf.reduce_mean(ae)
                    atan2 = tf.atan2(tf.sin(deviations), tf.cos(deviations))
                    loss = tf.square(atan2, name="loss")
                    mean_loss = tf.reduce_mean(loss)
                    loss_summary = tf.summary.scalar("loss", mean_loss)

                with tf.name_scope("loss2"):
                #     print(tf.shape(prediction))
                #     print(tf.shape(y_))
                    phi = prediction[:, :, 0]
                    phi0 = y_[:, :, 0]
                    psi = prediction[:, :, 1]
                    psi0 = y_[:,:, 1]
                    # cos_phi_diff = tf.square(tf.subtract(tf.cos(phi), tf.cos(phi0)))
                    # sin_phi_diff = tf.square(tf.subtract(tf.sin(phi), tf.sin(phi0)))
                    # cos_psi_diff = tf.square(tf.subtract(tf.cos(psi), tf.cos(psi0)))
                    # sin_psi_diff = tf.square(tf.subtract(tf.sin(psi), tf.sin(psi0)))
                    # phi_squared_sum = tf.add(cos_phi_diff, sin_phi_diff)
                    # psi_squared_sum = tf.add(cos_psi_diff, sin_psi_diff)
                    phi_diff = tf.reduce_sum(tf.squared_difference(phi, phi0))/2
                    psi_diff = tf.reduce_sum(tf.squared_difference(psi, psi0))/2
                    loss2 = tf.add(phi_diff, psi_diff)

                with tf.name_scope("mse"):
                    mse = tf.squared_difference(prediction, y_)
                    mse_summary = tf.summary.scalar("mse", mse)

                with tf.name_scope("l2_loss"):
                    l2_loss = beta * regularizers
                    if (use_l2):
                        loss = loss + l2_loss
                    loss = tf.reduce_mean(loss)
                    l2_summary = tf.summary.scalar("l2_loss", l2_loss)

                with tf.name_scope("train"):
                    # Use Adam optimizer
                    optimization = tf.train.AdamOptimizer(learning_rate).minimize(loss)
                # with tf.name_scope("accuracy"):
                #     correct_prediction = tf.equal(prediction, y)
                #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                #     tf.summary.scalar("accuracy", accuracy)

                summ = tf.summary.merge_all()

                print("Window size: " + str(window_size))
                print("Layers: " + str(layers_num))
                print("Channels: " +  str(channels_num))
                print("Beta: " + str(beta))
                print("Use L2: " + str(use_l2))
                print("Use Early stopping: " + str(use_early_stopping))
                sess = tf.InteractiveSession()
                init = tf.global_variables_initializer()
                sess.run(init)
                saver = tf.train.Saver()

                min_delta = 0.01
                plot_legend = []
                previous_epoch_min = 100
                min_validation_loss = 100
                for epoch in range(training_epochs):
                    train_session += 1
                    loss_arr = []
                    previous_batch_loss = 0.0
                    patience = 6
                    patience_cnt = 0

                    hparam = make_hparam_string(layers_num, channels_num, train_session)
                    writer = tf.summary.FileWriter(LOGDIR + hparam)
                    writer.add_graph(sess.graph)
                    total_batches = int(train_size/batch_size)
                    # Loop over all batches
                    for i in range(total_batches):
                        start_index = i * batch_size
                        stop_index = (i+1) * batch_size
                        batch_x = train_set[start_index:stop_index]
                        batch_y = train_y[start_index:stop_index]
                        # Run optimization op
                        # backprop and cost op (to get loss value)
                        if i % 5 == 0:
                            batch_predictions, l_summ, batch_loss = sess.run([prediction, loss_summary, loss], feed_dict={x: batch_x, y_: batch_y, keep_prob: dropout_keep_rate, keep_prob_input: 0.8})
                            writer.add_summary(l_summ, i+1)
                            loss_arr.append(batch_loss)
                            saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
                            # batch_predictions = np.apply_along_axis(convert_to_degrees, 2, batch_predictions)
                            batch_accuracy, batch_distr = calculate_accuracy(batch_predictions, batch_y)
                            # print('step %d, training accuracy %g' % (i, np.average(batch_accuracy)))
                            # early stopping
                            if(use_early_stopping):
                                if (epoch > 2 and i > total_batches / 2 and batch_loss < previous_epoch_min):
                                    previous_epoch_min = min(loss_arr)
                                    print("Early stopping!!")
                                    break
                        optimization.run(feed_dict={x: batch_x, y_: batch_y})
                    previous_epoch_min = min(loss_arr)
                    # Display logs per epoch step
                    if epoch % display_step == 0:
                        predictions, train_loss = sess.run([prediction,loss], feed_dict={x: train_set, y_: train_y, keep_prob: dropout_keep_rate, keep_prob_input: 0.8})
                        # predictions = np.apply_along_axis(convert_to_degrees, 2, predictions)
                        # plot_ramachandran(train_y, "Real values_"+str(epoch))
                        # raw_input()
                        train_accuracy, train_acc_distr = calculate_accuracy(predictions, train_y)
                        train_accuracy = np.average(train_accuracy)
                        crossvalidation_train_accuracy += train_accuracy
                        plot_legend.append('train_' + str(epoch))
                        # plot_loss(loss_arr)

                        # print("Training accuracy: ", \
                        #   "{:.6f}".format(train_accuracy))
                    if (epoch > training_epochs / 2):
                        valid_predictions, valid_loss, valid_mae = sess.run([prediction, loss, mae], feed_dict={x: test_set, y_: test_y})
                        # valid_predictions = np.apply_along_axis(convert_to_degrees, 2, valid_predictions)
                        valid_accuracy, valid_acc_distr = calculate_accuracy(valid_predictions, test_y)
                        valid_accuracy = np.average(valid_accuracy)
                        if (epoch >= training_epochs - 1):
                            if (valid_loss < min_validation_loss):
                                training_epochs += 1
                                print("INCREASING EPOCHS")
                            else:
                                crossvalidation_test_accuracy += valid_accuracy
                                crossvalidation_test_mae += valid_mae
                                for label in valid_acc_distr:
                                    crossvalidation_accuracy_distr[label] += valid_acc_distr[label]
                                print(crossvalidation_accuracy_distr)

                        if (epoch >= training_epochs - 2):
                            min_validation_loss = valid_loss
                            print(valid_acc_distr)
                            print("Validation accuracy: ", \
                                  "{:.6f}".format(valid_accuracy))


                    executed_epochs += 1
                # Test trained model
                test_predictions, test_summ, test_mae = sess.run([prediction, loss_summary, mae], feed_dict={x: test_set, y_: test_y})
                writer.add_summary(test_summ, i + 1)
                test_accuracy, test_acc_distr = calculate_accuracy(test_predictions, test_y)
                plot_ramachandran(test_predictions, "Predictions Fold "+str(test_session))
                plot_ramachandran(test_y, "Real values Fold "+str(test_session))

                # plot_legend.append('validation')

                print(test_acc_distr)
                # test_accuracy = np.average(test_accuracy)
                # crossvalidation_test_accuracy += test_accuracy
                # crossvalidation_test_mae += test_mae
                # print("Testing accuracy: ", \
                #       "{:.6f}".format(test_accuracy))
            for label in crossvalidation_accuracy_distr:
                crossvalidation_accuracy_distr[label] /= num_splits
            print(crossvalidation_accuracy_distr)
            # print("Final Testing DISTR: ", \
            #       "{:.6f}".format(crossvalidation_test_mae / num_splits))
            print("Final Testing MAE: ", \
                      "{:.6f}".format(crossvalidation_test_mae / num_splits))
            # print("Final Training accuracy: ", \
            #       "{:.6f}".format(crossvalidation_train_accuracy / (num_splits*training_epochs)))
            print("Final Test accuracy: ", \
                        "{:.6f}".format(crossvalidation_test_accuracy / num_splits))
            print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)

            # valid_predictions = sess.run(tf.argmax(prediction, 2), feed_dict={x: valid_x, y_: valid_y})
            # valid_labels = np.argmax(valid_y, 2)
            # valid_accuracy = calculate_accuracy(valid_predictions, valid_labels)
            # print("Validation accuracy: ", \
    #             "{:.6f}".format(valid_accuracy))