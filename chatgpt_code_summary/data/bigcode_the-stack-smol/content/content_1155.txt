# -*- coding: utf-8 -*-

import tensorflow as tf

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
from aquaman_net import AquamanNet

from utils import IMAGE_SIZE

EPOCHS = 1000
BATCH_SIZE = 4


def preproc(image_bytes):
    image_jpg = tf.image.decode_jpeg(image_bytes, channels=3)
    image_jpg = tf.image.resize_images(image_jpg, IMAGE_SIZE)
    image_jpg = tf.to_float(image_jpg) / 255.0
    image_jpg = tf.reshape(
        image_jpg, [IMAGE_SIZE[0], IMAGE_SIZE[1], 3], name="Reshape_Preproc")

    return image_jpg


def input_fn(tf_records_list, epochs=10, batch_size=8, n_frames=16):

    def _parse_proto(example_proto):
        parsed_dict = {
            "target": tf.FixedLenFeature((), tf.float32, default_value=0)
        }

        for i in range(n_frames):
            parsed_dict['frame_{}'.format(i)] = tf.FixedLenFeature(
                (), tf.string, default_value="")
        parsed_features = tf.parse_single_example(example_proto, parsed_dict)

        return parsed_features

    def _split_xy(feat_dict):
        target = tf.one_hot(tf.to_int32(
            feat_dict['target']), depth=2, dtype=tf.float32)

        input_frames = {}
        for i in range(n_frames):
            frame_id = 'frame_{}'.format(i)
            input_frames[frame_id] = feat_dict[frame_id]

        return input_frames, {'target': target}

    def _input_fn():
        dataset = tf.data.TFRecordDataset(
            tf_records_list, compression_type='GZIP')
        dataset = dataset.map(_parse_proto)
        dataset = dataset.map(_split_xy)
        dataset = dataset.shuffle(buffer_size=2 * batch_size)
        dataset = dataset.repeat(epochs)
        dataset = dataset.batch(batch_size)

        return dataset
    return _input_fn


def metrics(logits, labels):
    argmax_logits = tf.argmax(logits, axis=1)
    argmax_labels = tf.argmax(labels, axis=1)

    return {'accuracy': tf.metrics.accuracy(argmax_labels, argmax_logits)}


def get_serving_fn(window_size):
    input_tensor = {"frame_{}".format(i): tf.placeholder(
        dtype=tf.string, shape=[None]) for i in range(window_size)}
    return tf.estimator.export.build_raw_serving_input_receiver_fn(input_tensor)


def model_fn(n_frames):

    def _model_fn(features, labels, mode, params):

        input_tensors_list = []

        for i in range(n_frames):
            frame_id = 'frame_{}'.format(i)
            frame_tensor = tf.map_fn(preproc, features[frame_id], tf.float32)
            frame_tensor = tf.expand_dims(frame_tensor, axis=-1)
            frame_tensor = tf.transpose(frame_tensor, [0, 1, 2, 4, 3])
            print(frame_tensor)
            input_tensors_list.append(frame_tensor)

        input_tensor_stream = tf.concat(input_tensors_list, axis=3)
        print(input_tensor_stream)

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        logits = AquamanNet(input_tensor_stream, is_training, 2)

        # Loss, training and eval operations are not needed during inference.
        total_loss = None
        loss = None
        train_op = None
        eval_metric_ops = {}
        export_outputs = None

        prediction_dict = {'class': tf.argmax(
            logits, axis=1, name="predictions")}

        if mode != tf.estimator.ModeKeys.PREDICT:

            # IT IS VERY IMPORTANT TO RETRIEVE THE REGULARIZATION LOSSES
            reg_loss = tf.losses.get_regularization_loss()

            # This summary is automatically caught by the Estimator API
            tf.summary.scalar("Regularization_Loss", tensor=reg_loss)

            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=labels['target'], logits=logits)

            tf.summary.scalar("XEntropy_LOSS", tensor=loss)

            total_loss = loss + reg_loss

            learning_rate = tf.constant(1e-4, name='fixed_learning_rate')
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            vars_to_train = tf.trainable_variables()
            tf.logging.info("Variables to train: {}".format(vars_to_train))

            if is_training:
                # You DO must get this collection in order to perform updates on batch_norm variables
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.minimize(
                        loss=total_loss, global_step=tf.train.get_global_step(), var_list=vars_to_train)

            eval_metric_ops = metrics(logits, labels['target'])

        else:
            # pass
            export_outputs = {
                'logits': tf.estimator.export.PredictOutput(outputs=logits)}

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=prediction_dict,
            loss=total_loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            export_outputs=export_outputs)

    return _model_fn


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train-tf-list',
                        dest='train_tf_list',
                        type=str,
                        required=True)
    parser.add_argument('--test-tf-list',
                        dest='test_tf_list',
                        type=str,
                        required=True)
    parser.add_argument('--output-dir',
                        dest='output_dir',
                        type=str,
                        required=True)
    parser.add_argument('--window-size',
                        dest='window_size',
                        type=int,
                        required=True)
    args = parser.parse_args()

    tfrecord_list_train = args.train_tf_list.split(',')
    tfrecord_list_test = args.test_tf_list.split(',')

    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )

    run_config = tf.estimator.RunConfig(
        model_dir=args.output_dir,
        save_summary_steps=100,
        session_config=session_config,
        save_checkpoints_steps=100,
        save_checkpoints_secs=None,
        keep_checkpoint_max=1
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn(args.window_size),
        config=run_config
    )

    train_input_fn = input_fn(
        batch_size=BATCH_SIZE, tf_records_list=tfrecord_list_train, epochs=EPOCHS, n_frames=args.window_size)
    test_input_fn = input_fn(
        batch_size=BATCH_SIZE, tf_records_list=tfrecord_list_test, epochs=1, n_frames=args.window_size)

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=10000)

    # eval_steps = math.ceil(EVAL_SET_SIZE / FLAGS.batch_size)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=test_input_fn,
        # steps=eval_steps,
        start_delay_secs=60,
        throttle_secs=60)

    tf.estimator.train_and_evaluate(
        estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)

    estimator.export_savedmodel(
        export_dir_base=args.output_dir, serving_input_receiver_fn=get_serving_fn(args.window_size))
