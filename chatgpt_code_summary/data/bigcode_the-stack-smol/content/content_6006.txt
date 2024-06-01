"""
Training and validation method for arbitrary models.
"""
import io
import os
import sys
import time

from keras import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
import pandas as pd


plt.switch_backend('agg')


def train_and_evaluate(model, model_name, training_epochs, the_data):

    print('model.metrics_names: {}'.format(model.metrics_names))

    total_steps = training_epochs * the_data.get_training_mini_batches_per_epoch()
    training_index = pd.RangeIndex(start=0, stop=total_steps, name='Training Step')
    training_metrics_df = pd.DataFrame(
        data=np.zeros((total_steps, len(model.metrics_names))),
        columns=model.metrics_names,
        index=training_index)

    # evaluate the model on the dev set(s) after each epoch
    dev_index = pd.RangeIndex(start=0, stop=training_epochs, name='Epoch')
    dev_columns = pd.MultiIndex.from_product(
        iterables=(the_data.get_dev_set_names(), model.metrics_names),
        names=('dev set', 'metric'))
    dev_metrics_df = pd.DataFrame(
        data=np.zeros((training_epochs, len(the_data.get_dev_set_names()) * len(model.metrics_names))),
        columns=dev_columns,
        index=dev_index)

    print(dev_metrics_df.head())

    steps_per_epoch = the_data.get_training_mini_batches_per_epoch()

    # n counts number of training iterations
    n = 0
    t0 = time.time()
    ##with h5py.File(the_data.fp, 'r', libver='latest', swmr=True) as train_test_file:
    # train for all epochs
    t00 = time.time()
    ##for train_X, train_y, step, epoch in the_data.get_training_mini_batches(data_file=train_test_file, yield_state=True):
    for train_X, train_y, step, epoch in the_data.get_training_data_generator()(yield_state=True):
        if epoch > training_epochs:
            print('completed {} training epochs in {:5.2f}s'.format(training_epochs, time.time()-t0))
            break
        else:
            # train on one mini batch
            print('training on batch {} ({})'.format(step, steps_per_epoch))
            training_metrics = model.train_on_batch(train_X, train_y)
            training_metrics_df.loc[n, model.metrics_names] = training_metrics
            n += 1

        # look at performance on dev data after each epoch
        # re-plot the training and dev metrics after each epoch
        if step == steps_per_epoch:
            print('completed training epoch {} in {:5.2f}s'.format(epoch, time.time()-t00))
            print('{} steps per epoch'.format(steps_per_epoch))
            print('{:5.2f}s per step'.format((time.time()-t00)/steps_per_epoch))
            print(training_metrics_df.loc[n-2:n])
            t00 = time.time()
            print('evaluate the model on the dev set(s)')

            #evaluate_dev_sets(epoch=epoch, model=model, the_data=the_data, train_test_file=train_test_file, dev_metrics_df=dev_metrics_df)
            evaluate_dev_sets(epoch=epoch, model=model, the_data=the_data, dev_metrics_df=dev_metrics_df)

            plot_training_and_dev_metrics(
                training_metrics_df,
                dev_metrics_df,
                model_name=model_name,
                steps_per_epoch=steps_per_epoch,
                epoch_count=training_epochs,
                output_fp=model_name + '.pdf')

    return training_metrics_df, dev_metrics_df


def evaluate_dev_sets(epoch, model, the_data, dev_metrics_df):

    for dev_steps, dev_set_name, dev_generator in the_data.get_dev_generators():
        sys.stdout.write('.')
        # print('dev set: "{}"'.format(dev_set_name))
        # print('  dev steps: {}'.format(dev_steps))
        dev_metrics = model.evaluate_generator(generator=dev_generator, steps=dev_steps)
        dev_metrics_df.loc[epoch - 1, (dev_set_name, model.metrics_names)] = dev_metrics
    sys.stdout.write('\n')
    print('dev metrics:\n{}'.format(dev_metrics_df.loc[epoch - 1]))


def build_layer(model_name, layer_type, kwargs):
    if layer_type == 'Dense':
        model_name.write('_dns_{}'.format(kwargs['units']))
        if 'kernel_regularizer' in kwargs:
            # the l2 field is a ndarray with shape ()
            # indexing with [] gives error 'too many indices'
            # the item() method is the first way I found to extract the float value from l2
            model_name.write('_l2_{:6.4f}'.format(kwargs['kernel_regularizer'].l2.item()))
        layer = Dense(**kwargs)
    elif layer_type == 'Dropout':
        model_name.write('_drp_{:3.2f}'.format(kwargs['rate']))
        layer = Dropout(**kwargs)
    elif layer_type == 'BatchNormalization':
        model_name.write('_bn')
        layer = BatchNormalization(**kwargs)
    else:
        raise Exception()

    return layer


def build_model(layers, model=None, input_dim=None):
    """
    Build and return a Sequential model with Dense layers given by the layers argument.

    Arguments
        model      (keras.Sequential) model to which layers will be added
        input_dim  (int) dimension of input
        layers     (tuple) sequence of 2-ples, one per layer, such as ((64, 'relu'), (64, 'relu'), (1, 'sigmoid'))

    Return
        model_name (str) a name for the model
        model      (Model) a compiled model
    """
    if model is None:
        model = Sequential()

    model_name = io.StringIO()
    layer_type, kwargs = layers[0]
    if input_dim is None:
        pass
    else:
        kwargs['input_dim'] = input_dim

    for layer_type, kwargs in layers:
        layer = build_layer(model_name, layer_type, kwargs)
        model.add(layer)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # trim the leading '_' from the model name - lazy!
    return model_name.getvalue()[1:], model


def plot_training_and_dev_metrics(training_metrics_df, dev_metrics_df, model_name, steps_per_epoch, epoch_count, output_fp):
    # generate network-specific accuracy and loss keys
    output_dp, output_filename = os.path.split(output_fp)
    output_basename, output_ext = os.path.splitext(output_filename)

    ##separate_plots_fp = os.path.join(output_dp, output_basename + '_separate' + output_ext)

    ##sorted_training_history_list = sorted(training_history_list, key=lambda h: h[2]['val_acc'][-1], reverse=True)

    with PdfPages(output_fp) as pdfpages:
        #for model_name, layers, history, t in sorted_training_history_list:
            #training_accuracy_loss = {}
            #validation_accuracy_loss = {}

            #training_accuracy_loss['acc ' + model_name] = history['acc']
            #training_accuracy_loss['loss ' + model_name] = history['loss']
            #validation_accuracy_loss['val_acc ' + model_name] = history['val_acc']
            #validation_accuracy_loss['val_loss ' + model_name] = history['val_loss']

            #training_df = pd.DataFrame(
            #    data=training_accuracy_loss,
            #    index=[b + 1 for b in range(epoch_count * batches_per_epoch)])
            #training_df.index.name = 'batch'

            #validation_df = pd.DataFrame(
            #    data=validation_accuracy_loss,
            #    index=[(e + 1) * batches_per_epoch for e in range(epoch_count)])
            #validation_df.index.name = 'batch'

            fig, ax1 = plt.subplots()
            legend = []
            #for loss_column in [column for column in training_df.columns if 'loss' in column and model_name in column]:
            #for training_metric_column in training_metrics_df.columns:
            #print('training metric column: {}'.format(training_metric_column))
            ax1.plot(training_metrics_df.index, training_metrics_df.loc[:, 'loss'], color='tab:blue', alpha=0.8)
            legend.append('training loss')
            #for loss_column in [column for column in validation_df.columns if
            #                    'loss' in column and model_name in column]:
            #    print('validation loss column: {}'.format(loss_column))
            #    ax1.plot(validation_df.index, validation_df.loc[:, loss_column], color='tab:orange', alpha=0.8)
            #    legend.append('val_loss')
            ax1.set_xlabel('epoch')
            tick_spacing = steps_per_epoch
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax1.set_xticklabels([0] + list(range(epoch_count+1)))
            ax1.set_ylabel('loss')
            ax1.legend(legend, loc='lower left')

            ax2 = ax1.twinx()
            legend = []
            #for acc_column in [column for column in training_metrics_df.columns if 'acc' in column]:
            #print('training acc column: {}'.format(acc_column))
            ax2.plot(training_metrics_df.index, training_metrics_df.loc[:, 'acc'], color='tab:purple', alpha=0.8)
            legend.append('training acc')
            for dev_acc_column in [column for column in dev_metrics_df.columns if 'acc' in column]:
                print('validation acc column: {}'.format(dev_acc_column))
                ax2.plot([steps_per_epoch * (n + 1) for n in dev_metrics_df.index], dev_metrics_df.loc[:, dev_acc_column], alpha=0.8)
                legend.append(dev_acc_column)
            ax2.set_title('Training and Development Metrics\n{}'.format(model_name))
            ax2.set_ylim(0.0, 1.0)
            ax2.set_ylabel('accuracy')
            print(legend)
            ax2.legend(legend, loc='lower right')

            pdfpages.savefig()


