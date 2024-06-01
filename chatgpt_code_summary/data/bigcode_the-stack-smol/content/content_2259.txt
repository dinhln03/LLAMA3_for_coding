import os
import json
import datetime

import numpy as np
from matplotlib import pyplot as plt


class MetaLogger(object):

    def __init__(self, meta_config, config, task_directory, load_directory=None, load_epoch=None):
        self.results_directory = os.path.join('meta_results', str(datetime.datetime.now()))
        self.results = {
            'task_directory': task_directory,
            'load_directory': load_directory,
            'load_epoch': load_epoch,
            'train_losses': [],
            'train_accuracies': [],
            'validation_losses': [],
            'validation_accuracies': [],
            'baseline_test_loss': 0,
            'baseline_test_accuracy': 0,
            'sgd_test_loss': 0,
            'sgd_test_accuracy': 0,
            'adam_test_loss': 0,
            'adam_test_accuracy': 0,
            'meta_optimizer_test_loss': 0,
            'meta_optimizer_test_accuracy': 0,
            'config': config,
            'meta_config': meta_config
        }

    def load(self, file_path):
        self.results_directory, _ = os.path.split(file_path)
        with open(file_path, 'r') as file_obj:
            self.results = json.load(file_obj)

    def log(self):
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)
        with open('{}/results.json'.format(self.results_directory), 'w') as file_obj:
            json.dump(self.results, file_obj, indent=4)

    def plot(self):
        plt.figure()
        plt.title('Loss')
        plt.xlabel('Meta Epochs')
        plt.ylabel('Loss')
        plt.xticks(np.arange(0, len(self.results['train_losses']) * .125, .25))
        plt.plot(np.arange(.125, (len(self.results['train_losses']) + 1) * .125, .125), self.results['train_losses'], label='train')
        plt.plot(np.arange(.125, (len(self.results['validation_losses']) + 1) * .125, .125), self.results['validation_losses'], label='validation')
        plt.legend()
        plt.savefig('{}/loss.pdf'.format(self.results_directory))
        plt.close()

        plt.figure()
        plt.title('Accuracy')
        plt.xlabel('Meta Epochs')
        plt.ylabel('Accuracy')
        plt.xticks(np.arange(0, len(self.results['train_accuracies']) * .125, .25))
        plt.plot(np.arange(.125, (len(self.results['train_accuracies']) + 1) * .125, .125), self.results['train_accuracies'], label='train')
        plt.plot(np.arange(.125, (len(self.results['validation_accuracies']) + 1) * .125, .125), self.results['validation_accuracies'], label='validation')
        plt.legend()
        plt.savefig('{}/accuracy.pdf'.format(self.results_directory))
        plt.close()

        plt.figure()
        plt.title('Test Losses')
        plt.ylabel('Mean Test Loss')
        x_labels = ('Baseline', 'SGD', 'Adam', 'Meta Optimizer')
        x_pos = np.arange(len(x_labels))
        performance = [self.results['{}_test_loss'.format('_'.join(label.lower().split(' ')))] for label in x_labels]
        plt.bar(x_pos, performance, align='center', alpha=0.5)
        plt.xticks(x_pos, x_labels)
        plt.savefig('{}/test_loss.pdf'.format(self.results_directory))
        plt.close()

        plt.figure()
        plt.title('Test Accuracies')
        plt.ylabel('Mean Test Accuracy')
        x_labels = ('Baseline', 'SGD', 'Adam', 'Meta Optimizer')
        x_pos = np.arange(len(x_labels))
        performance = [self.results['{}_test_accuracy'.format('_'.join(label.lower().split(' ')))] for label in x_labels]
        plt.bar(x_pos, performance, align='center', alpha=0.5)
        plt.xticks(x_pos, x_labels)
        plt.savefig('{}/test_accuracy.pdf'.format(self.results_directory))
        plt.close()
