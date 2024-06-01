import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from net.tf_net import \
    calculate_accuracy, calculate_loss, \
    create_simple_cnn_model, optimize_weights
from net.keras_net import simple_cnn


def train_keras(batch_size, epochs, n_classes):
    # x_train returns data with shape (60,000,28,28)
    # y_train returns data with shape (60,000,)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # add one dimension for color chanel (only gray values)
    x_train = x_train.reshape(x_train.shape[0], image_height, image_width, 1)
    x_test = x_test.reshape(x_test.shape[0], image_height, image_width, 1)

    # define input shape of image
    input_shape = (image_height, image_width, 1)

    # convert tensor to float
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # normalize data: divide by 255 (max color value) to receive values between 0 and 1
    x_train /= 255
    x_test /= 255

    # one-hot encoding: converts into array of length 'n_classes' and sets one where true
    # e.g. label = 5 y_train[4]=1, rest is 0
    y_train = tf.keras.utils.to_categorical(y_train, n_classes)
    y_test = tf.keras.utils.to_categorical(y_test, n_classes)

    simple_cnn_model = simple_cnn(input_shape)

    simple_cnn_model.fit(x_train, y_train, batch_size,
                         epochs, (x_test, y_test))

    train_loss, train_accuracy = simple_cnn_model.evaluate(
        x_train, y_train, verbose=0)
    print('Train data loss:', train_loss)
    print('Train data accuracy:', train_accuracy)

    test_loss, test_accuracy = simple_cnn_model.evaluate(
        x_test, y_test, verbose=0)
    print('Test data loss:', test_loss)
    print('Test data accuracy:', test_accuracy)


def train_tensorflow(batch_size, epochs, n_classes):
    mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
    test_images, test_labels = mnist_data.test.images, mnist_data.test.labels
    input_size = 784

    # declare placeholder
    x_input = tf.placeholder(tf.float32, shape=[None, input_size])
    y_input = tf.placeholder(tf.float32, shape=[None, n_classes])
    # if test set dropout to false
    bool_dropout = tf.placeholder(tf.bool)

    # create neural net and receive logits
    logits = create_simple_cnn_model(x_input, y_input, bool_dropout)
    # calculate loss, optimize weights and calculate accuracy
    loss_operation = calculate_loss(logits, y_input)
    optimizer = optimize_weights(loss_operation)
    accuracy_operation = calculate_accuracy(logits, y_input)

    # start training
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # merge all summary for tensorboard
    merged_summary_operation = tf.summary.merge_all()
    train_summary_writer = tf.summary.FileWriter('/tmp/train', session.graph)
    test_summary_writer = tf.summary.FileWriter('/tmp/test')

    for batch_n in range(epochs):
        mnist_batch = mnist_data.train.next_batch(batch_size)
        train_images, train_labels = mnist_batch[0], mnist_batch[1]

        _, merged_summary = session.run([optimizer, merged_summary_operation],
                                        feed_dict={
            x_input: train_images,
            y_input: train_labels,
            bool_dropout: True
        })

        train_summary_writer.add_summary(merged_summary, batch_n)

        if batch_n % 10 == 0:
            merged_summary, _ = session.run([merged_summary_operation, accuracy_operation],
                                            feed_dict={
                x_input: test_images,
                y_input: test_labels,
                bool_dropout: False
            })

            test_summary_writer.add_summary(merged_summary, batch_n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a simple neural net to recognize number images from the MNIST dataset and appy the correct labeld')
    parser.add_argument('--epochs', default=200,
                        help='Amount of batches the net trains on')
    parser.add_argument('--batch_size', default=100,
                        help='Number of training samples inside one batch')
    parser.add_argument('--tf', default=True,
                        help='Tensorflow or Keras implementation')
    args = parser.parse_args()

    if(args.tf):
        train_tensorflow(args.batch_size, args.epochs, 10)
    else:
        train_keras(args.batch_size, args.epochs, args.n_classes)
