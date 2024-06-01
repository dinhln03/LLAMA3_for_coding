import matplotlib
matplotlib.use('Agg')
import os
from os.path import join
import argparse
import torch
import numpy as np
import pickle
import sys
import datetime
sys.path.append('./utils')

from torch import optim
from torch import nn
from torch import multiprocessing
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
from utils.builders import SingleViewDepthTripletBuilder, MultiViewDepthTripletBuilder, MultiViewTripletBuilder, SingleViewTripletBuilder
from utils.builder_utils import distance, Logger, ensure_folder, collate_fn, time_stamped
from utils.vocabulary import Vocabulary
from ipdb import set_trace
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import transforms
import torchvision.utils as vutils
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from shutil import copy2
import importlib
from pyquaternion import Quaternion

from models.pose_predictor_euler_crop import define_model
from utils.plot_utils import plot_mean
from utils.rot_utils_old import create_rot_from_vector, rotationMatrixToEulerAngles, \
                            isRotationMatrix, eulerAnglesToRotationMatrix, \
                            norm_sincos, sincos2rotm
from utils.network_utils import loss_rotation, loss_euler_reparametrize, loss_axisangle, batch_size, apply,\
                                    loss_quat, loss_quat_single, euler_XYZ_to_reparam, loss_quat_huber
from utils.plot_utils import plot_mean

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= "1,2,3"
  
IMAGE_SIZE = (299, 299)
NUM_VIEWS = 1
SAMPLE_SIZE = 40
VAL_SEQS =5
TRAIN_SEQS_PER_EPOCH = 80
LOSS_FN = loss_euler_reparametrize

EXP_ROOT_DIR = '/media/hdd/msieb/data/tcn_data/experiments'
sys.path.append(EXP_ROOT_DIR)

class Trainer(object):
    def __init__(self, use_cuda, load_model, model_folder, train_directory, validation_directory, builder, loss_fn, args, multi_gpu=True):
        self.use_cuda = use_cuda
        self.load_model = load_model
        self.model_folder = model_folder
        self.validation_directory = validation_directory
        self.train_directory = train_directory
        self.args = args

        self.builder = builder
        self.loss_fn = loss_fn
        self.logdir = join(model_folder, 'logs')
        self.writer = SummaryWriter(self.logdir)
        self.logger = Logger(self.args.log_file)
        self.itr = 0

        # Create Model
        self.model = self.create_model()
        if multi_gpu:
            self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))

        # Build validation set
        validation_builder = builder(self.args.n_views, validation_directory, IMAGE_SIZE, self.args, toRot=True, sample_size=SAMPLE_SIZE)
        validation_set = [validation_builder.build_set() for i in range(VAL_SEQS)]
        validation_set = ConcatDataset(validation_set)
        self.len_validation_set = len(validation_set)
        del validation_builder
        self.validation_loader = DataLoader(
                        validation_set, 
                        batch_size=8, 
                        shuffle=False, 
                        pin_memory=self.use_cuda,
                        )
        self.validation_calls = 0
        # Build Training Set
        self.triplet_builder = builder(self.args.n_views, \
            train_directory, IMAGE_SIZE, self.args, toRot=True, sample_size=SAMPLE_SIZE)
        self.training_queue = multiprocessing.Queue(1)
        dataset_builder_process = multiprocessing.Process(target=self.build_set, args=(self.training_queue, self.triplet_builder, self.logger), daemon=True)
        dataset_builder_process.start()

        # Get Logger
   

        # Model specific setup
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr_start, momentum=0.9)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
        # This will diminish the learning rate at the milestones ///// 0.1, 0.01, 0.001 if not using automized scheduler
        self.learning_rate_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        # self.criterion = nn.CrossEntropyLoss()

    def train(self):

        trn_losses_ = []
        val_losses_= []
        val_acc_ = []
        trn_acc_ = []


        for epoch in range(self.args.start_epoch, self.args.start_epoch + self.args.epochs):
            print("=" * 20)
            self.logger.info("Starting epoch: {0} ".format(epoch))

            dataset = self.training_queue.get()
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=self.args.minibatch_size, # batch_size(epoch, self.args.max_minibatch_size),
                shuffle=True,
                pin_memory=self.use_cuda,
            )
            
            train_embedding_features_buffer = []
            train_images_buffer = []
            train_labels = []
            correct = 0

            for _ in range(0, 1):
                losses = []

                for minibatch in data_loader:
                    if self.use_cuda:
                        anchor_frames = minibatch[0].cuda()
                        #anchor_euler_reparam = minibatch[1].cuda() # load as 3x3 rotation matrix
                        anchor_quats = minibatch[1].cuda() # load as 3x3 rotation matrix
                    # frames = Variable(minibatch)
                    loss, a_pred = self.loss_fn(self.model, anchor_frames, anchor_quats)
                    losses.append(loss.data.cpu().numpy()) 
                    correct += (torch.norm(a_pred - anchor_quats, 2) < 1).data.cpu().numpy().sum()                    # print(gradcheck(loss_fn, (tcn, minibatch,)))     
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Add embeddings
                    train_labels.append(anchor_quats)
                    train_embedding_features_buffer.append(anchor_quats)
                    train_images_buffer.append(anchor_frames)
            print("logging to {}".format(self.logdir))

            self.writer.add_scalar('data/train_loss', np.mean(losses), self.itr)
            self.writer.add_scalar('data/train_correct', correct / len(data_loader), self.itr)
            self.itr += 1  
            trn_losses_.append(np.mean(losses))
            self.logger.info('train loss: ', np.mean(losses))
            self.logger.info("Training score correct  {correct}/{total}".format(
            correct=correct,
            total=len(data_loader)
            ))
            trn_acc_.append(correct)

            self.writer.add_image('frame_1', minibatch[0][0], self.itr)
            # self.writer.add_image('pose1', str(minibatch[1][0].data.detach().cpu().numpy()), self.itr)
            self.writer.add_image('frame_2', minibatch[0][1], self.itr)
            # self.writer.add_image('pose_2', str(minibatch[1][1].data.detach().cpu().numpy()), self.itr)           
            self.writer.add_image('frame_3', minibatch[0][2], self.itr)
            # self.writer.add_image('pose_3', str(minibatch[1][2].data.detach().cpu().numpy()), self.itr)
            self.writer.add_image('frame_4', minibatch[0][3], self.itr)
            # self.writer.add_image('pose_4', str(minibatch[1][3].data.detach().cpu().numpy()), self.itr)
            # Get embeddings
            features = torch.cat(train_embedding_features_buffer[:30]).squeeze_()
            labels = torch.cat(train_labels[:30]).squeeze_()
            # features = train_embedding_features_buffer.view(train_embedding_features_buffer.shape[0]*train_embedding_features_buffer.shape[1], -1)
            # label = torch.Tensor(np.asarray(label_buffer))
            images = torch.cat(train_images_buffer[:30]).squeeze_()#/255.0, [0, 3, 1, 2]
            self.writer.add_embedding(features, metadata=labels, label_img=images, global_step=epoch)
            
            if epoch % 1 == 0:
                loss, correct  = self.validate()
                self.learning_rate_scheduler.step(loss)
                val_losses_.append(loss)
                val_acc_.append(correct)

            if epoch % self.args.save_every == 0 and epoch != 0:
                self.logger.info('Saving model.')
                self.save_model(self.model, self.model_filename(self.args.model_name, epoch), join(self.model_folder, 'weight_files'))
                print("logging to {}".format(self.logdir))

            plot_mean(trn_losses_, self.model_folder, 'train_loss')
            plot_mean(val_losses_, self.model_folder, 'validation_loss')
            plot_mean(trn_acc_, self.model_folder, 'train_acc')
            plot_mean(val_acc_, self.model_folder, 'validation_accuracy')
            # plot_mean(val_acc_no_margin_, self.model_folder, 'validation_accuracy_no_margin')

    def validate(self):
        # Run model on validation data and log results
        correct = 0
        losses = []
        for minibatch in self.validation_loader:
            if self.use_cuda:
                anchor_frames = minibatch[0].cuda()
                #anchor_euler_reparam = minibatch[1].cuda() # load as 3x3 rotation matrix
                anchor_quats = minibatch[1].cuda() # load as 3x3 rotation matrix
            loss, a_pred = self.loss_fn(self.model, anchor_frames, anchor_quats)
            losses.append(loss.data.cpu().numpy())
            correct += (torch.norm(a_pred - anchor_quats, 2) < 0.1).data.cpu().numpy().sum()

        self.writer.add_scalar('data/valid_loss', np.mean(losses), self.validation_calls)
        self.writer.add_scalar('data/validation_correct', correct / self.len_validation_set, self.validation_calls)

        self.validation_calls += 1
        loss = np.mean(losses)
        self.logger.info("Validation score correct  {correct}/{total}".format(
            correct=correct,
            total=self.len_validation_set
        ))
        self.logger.info('val loss: ',loss)
        return loss, correct
    

    def model_filename(self, model_name, epoch):
        return "{model_name}-epoch-{epoch}.pk".format(model_name=model_name, epoch=epoch)

    def save_model(self, model, filename, model_folder):
        ensure_folder(model_folder)
        model_path = os.path.join(model_folder, filename)
        torch.save(model.state_dict(), model_path)


    def build_set(self, queue, triplet_builder, log):
        while 1:
            datasets = []
            for i in range(TRAIN_SEQS_PER_EPOCH):
                dataset = triplet_builder.build_set()
                datasets.append(dataset)
            dataset = ConcatDataset(datasets)
            # log.info('Created {0} triplets'.format(len(dataset)))
            queue.put(dataset)

    def create_model(self):
        model = define_model(pretrained=True)
        # model = PosNet()
        if self.load_model:
            model_path = os.path.join(
                self.model_folder,
                self.load_model
            )
            # map_location allows us to load models trained on cuda to cpu.
            model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

        if self.use_cuda:
            model = model.cuda()
        return model

    def batch_size(self, epoch, max_size):
        exponent = epoch // 100
        return min(max(2 ** (exponent), 2), max_size)

def main(args):
    # module = importlib.import_module(args.exp_name + '.config')
    # conf = getattr(module, 'Config_Isaac_Server')()
    # EXP_DIR = conf.EXP_DIR
    # MODEL_FOLDER = conf.MODEL_FOLDER


    # GPU Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()

    # Load model
    model_folder = join(EXP_ROOT_DIR, args.exp_name, 'trained_models', args.run_name, time_stamped())
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Get data loader builder and loss function
    builder = getattr(importlib.import_module('utils.builders'), args.builder)
    loss_fn = LOSS_FN

    # Define train and validation directories
    train_directory = join(EXP_ROOT_DIR, args.exp_name, 'videos/train/') 
    validation_directory = join(EXP_ROOT_DIR, args.exp_name, 'videos/valid/') 

    # Copies of executed config
    if not os.path.exists('/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/experiments'):
        os.makedirs('/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/experiments')
    copy2('/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/train_tcn_no_captions.py', model_folder)
    copy2('/'.join(os.path.realpath(__file__).split('/')[:-2]) + '/gps-lfd' + '/config.py', model_folder)
            
    # Build training class
    trainer = Trainer(use_cuda, args.load_model, model_folder, train_directory, validation_directory, builder, loss_fn, args) 
    trainer.train()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--load-model', type=str, required=False)
   
    parser.add_argument('--minibatch-size', type=int, default=8)
    parser.add_argument('--model-name', type=str, default='tcn')
    parser.add_argument('--log-file', type=str, default='./out.log')
    parser.add_argument('--lr-start', type=float, default=0.001)
    parser.add_argument('--n-views', type=int, default=NUM_VIEWS)
    parser.add_argument('--alpha', type=float, default=0.01, help='weighing factor of language loss to triplet loss')

    # Model parameters
   
    # Path parameters
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--run-name', type=str, required=True)
    parser.add_argument('--builder', type=str, required=True)

    args = parser.parse_args()
    print(args)

    main(args)
