# author rovo98

import os

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

from model_data_input import load_processed_dataset
from models.fdconv1d_lstm.model import build_fdconv1d_lstm
from models.utils.misc import running_timer
from models.utils.misc import plot_training_history

# filter warning logs of tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# enable memory growth for every GPU.
# Using GPU devices to train the models is recommended.
# uncomment the following several lines of code to disable forcing using GPU.
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, 'Not enough GPU hardware available'
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)


# noinspection DuplicatedCode
@running_timer
def train_model(epochs=10,
                batch_size=32,
                training_verbose=1,
                print_model_summary=False,
                using_validation=False,
                validation_split=0.2,
                plot_history_data=False,
                history_fig_name='default',
                plot_model_arch=False,
                plot_model_name='default',
                save_model=False,
                save_model_name='default'):
    # num_of_faulty_type = 3
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-02-22 20:34:10_czE4OmZzNDphczE2OmZlczI=_processed_logs_rnn', num_of_faulty_type,
    #     location='../../dataset', for_rnn=True)
    #
    # num_of_faulty_type = 5
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2019-12-28 00:46:37_czc1OmZzNzphczE1OmZlczQ=_processed_logs', num_of_faulty_type,
    #     location='../../dataset')

    # 1. single faulty mode(small state size): short logs (10 - 50)
    num_of_faulty_type = 3
    train_x, train_y, test_x, test_y = load_processed_dataset(
        '2020-03-17 15:55:22_czE4OmZzNDphczE2OmZlczI=_processed_logs', num_of_faulty_type,
        location='../../dataset')
    # 2. single faulty mode(small state size): long logs (60 - 100)
    # num_of_faulty_type = 3
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-17 16:00:22_czE4OmZzNDphczE2OmZlczI=_processed_logs_b', num_of_faulty_type,
    #     location='../../dataset')

    # 3. single faulty mode(big state size): short logs (10 - 50)
    # num_of_faulty_type = 5
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-17 16:16:04_czgwOmZzODphczE4OmZlczQ=_processed_logs', num_of_faulty_type,
    #     location='../../dataset')
    # 4. single faulty mode(big state size): long logs (60 - 100)
    # num_of_faulty_type = 5
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-19 17:09:05_czgwOmZzODphczE4OmZlczQ=_processed_logs_b_rg', num_of_faulty_type,
    #     location='../../dataset')

    # 5. multi faulty mode (small state size): short logs
    # num_of_faulty_type = 4
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-17 16:34:50_czE3OmZzNDphczE0OmZlczI=_processed_logs', num_of_faulty_type,
    #     location='../../dataset')

    # 6. multi faulty mode (small state size): long logs
    # num_of_faulty_type = 4
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-17 16:36:40_czE3OmZzNDphczE0OmZlczI=_processed_logs_b', num_of_faulty_type,
    #     location='../../dataset')

    # 7. multi faulty mode (big state size): short logs
    # num_of_faulty_type = 16
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-17 16:40:03_czgwOmZzODphczIwOmZlczQ=_processed_logs', num_of_faulty_type,
    #     location='../../dataset')
    # 8. multi faulty mode (big state size): long logs
    # num_of_faulty_type = 16
    # train_x, train_y, test_x, test_y = load_processed_dataset(
    #     '2020-03-17 16:41:29_czgwOmZzODphczIwOmZlczQ=_processed_logs_b', num_of_faulty_type,
    #     location='../../dataset')

    n_timesteps, n_features = train_x.shape[1], train_x.shape[2]

    # building the model.
    model = build_fdconv1d_lstm((n_timesteps, n_features), num_of_faulty_type, kernel_size=31)

    # print out the model summary
    if print_model_summary:
        model.summary()

    # plot and save the model architecture.
    if plot_model_arch:
        plot_model(model, to_file=plot_model_name, show_shapes=True)

    # fit network
    if plot_history_data:
        history = model.fit(x=[train_x, train_x], y=train_y, epochs=epochs, batch_size=batch_size,
                            verbose=training_verbose, validation_split=validation_split)
        plot_training_history(history, 'fdconv1d-lstm', history_fig_name, '../exper_imgs')
    elif using_validation:
        es = EarlyStopping('val_categorical_accuracy', 1e-4, 3, 1, 'max')
        history = model.fit(x=[train_x, train_x], y=train_y, epochs=epochs, batch_size=batch_size,
                            verbose=training_verbose, validation_split=validation_split, callbacks=[es])
        plot_training_history(history, 'fdconv1d-lstm', history_fig_name, '../exper_imgs')
    else:
        model.fit(x=[train_x, train_x], y=train_y, epochs=epochs, batch_size=batch_size, verbose=training_verbose)

    _, accuracy = model.evaluate(x=[test_x, test_x], y=test_y, batch_size=batch_size, verbose=0)

    # saving the model
    if save_model:
        model.save(save_model_name)
        print('>>> model saved: {}'.format(save_model_name))

    print('\n>>> Accuracy on testing given testing dataset: {}'.format(accuracy * 100))


# Driver the program to test the methods above.
if __name__ == '__main__':
    train_model(50,
                print_model_summary=True,
                using_validation=True,
                history_fig_name='fdConv1d-lstm_czE4OmZzNDphczE2OmZlczI=_small.png',
                save_model=True,
                save_model_name='../trained_saved/fdConv1d-lstm_czE4OmZzNDphczE2OmZlczI=_small.h5')
