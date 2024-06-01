import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.PCDLoader import *


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=12, type=int, help='training on real dataset')
    parser.add_argument('--epoch', default=20, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_sparse_point', type=int, default=50, help='Point Number for domain loss')
    parser.add_argument('--random_choose_sparse', type=bool, default=False, help='Random select num_sparse_point from [10,20,30,40,50]')
    parser.add_argument('--SO3_Rotation', action='store_true', default=False, help='arbitrary rotation in SO3')
    parser.add_argument('--DA_method', type=str, default="multi_coral_mmd", help='choose the DA loss function')
    parser.add_argument('--alpha', type=float, default=10, help='set the value of classification loss')
    parser.add_argument('--lamda', type=float, default=10, help='set the value of CORAL loss')
    parser.add_argument('--beta', type=float, default=10, help='set the value of MMD loss')
    return parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, num_class=12):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, data in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = data['pointcloud'].to(device).float(), data['category'].to(device)

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    visual_data_path = 'data/visual_data_pcd/'
    tactile_data_path = 'data/tactile_pcd_10_sampled_21.02/'



    train_dataset = PCDPointCloudData(visual_data_path,
                                      folder='Train',
                                      sample_method='Voxel',
                                      num_point=args.num_point,
                                      sample=True,
                                      rotation=False,
                                      est_normal=args.use_normals)

    test_dataset = PCDPointCloudData(visual_data_path,
                                     folder='Test',
                                     sample_method='Voxel',
                                     num_point=args.num_point,
                                     sample=True,
                                     rotation=False,
                                     est_normal=args.use_normals)


    if args.random_choose_sparse is True:
        raise NotImplementedError("Function Not Implemented") # Not Implement
        # domain_adaptation_dataset = PCDPointCloudData(tactile_data_path, folder='Train',
        #                                               random_num=True,
        #                                               list_num_point=[10,20,30,40,50])
    else:
        domain_adaptation_dataset = PCDPointCloudData(tactile_data_path,
                                                      folder='Train',
                                                      sample_method='Voxel',
                                                      num_point=args.num_sparse_point,
                                                      sample=True,
                                                      rotation=False,
                                                      est_normal=args.use_normals)

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    domainAdaptationDataLoader = torch.utils.data.DataLoader(domain_adaptation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''Output middle layers'''
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation [name] = output.detach()
        return hook

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet_cls.py', str(exp_dir))
    shutil.copy('data_utils/PCDLoader.py', str(exp_dir))
    shutil.copy('./train_realMulti-DA-Loss_classification.py', str(exp_dir))

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    if args.DA_method == "coral":
        criterion_DA = model.get_coral_loss(DA_alpha=args.alpha, DA_lamda=args.lamda)
    elif args.DA_method == "mmd":
        criterion_DA = model.get_mmd_loss(DA_alpha=args.alpha, DA_lamda=args.lamda)
    elif args.DA_method == "coral_mmd":
        criterion_DA = model.get_coral_mmd_loss(DA_alpha=args.alpha, DA_beta=args.beta,
                                                DA_lamda=args.lamda)
    elif args.DA_method == "multi_coral_mmd":
        criterion_DA = model.get_multiLayer_loss(DA_alpha=args.alpha, DA_beta=args.beta,
                                                 DA_lamda=args.lamda)
    else:
        raise NameError("Wrong input for DA method name!")

    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        criterion_DA = criterion_DA.cuda()

    # Load pretrained model with real dataset
    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    try:
        min_loss = checkpoint['loss']
        log_string('Loading model with DA loss %f' % min_loss)
    except:
        log_string('No DA loss found in the model')
        min_loss = 10000.0


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    running_loss = 0.0

    '''TRANING'''
    logger.info('Start training...')
    end_epoch = start_epoch + args.epoch
    print("start epoch: ", start_epoch)
    print("end epoch: ", end_epoch)
    for epoch in range(start_epoch, end_epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, end_epoch))
        mean_correct = []
        # Freeze Conv
        for name, param in classifier.named_parameters():
            if "feat" in name:
                param.requires_grad = False


        scheduler.step()
        for batch_id, (data, data_DA) in tqdm(
                enumerate(zip(trainDataLoader,domainAdaptationDataLoader), 0),
                total=len(trainDataLoader),
                smoothing=0.9):

            optimizer.zero_grad()
            points, target = data['pointcloud'].to(device).float(), data['category'].to(device)
            points_DA = data_DA['pointcloud'].to(device).float()

            points = points.data.cpu().numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            points_DA = points_DA.data.cpu().numpy()
            points_DA = provider.random_point_dropout(points_DA)
            points_DA[:, :, 0:3] = provider.random_scale_point_cloud(points_DA[:, :, 0:3])
            points_DA[:, :, 0:3] = provider.shift_point_cloud(points_DA[:, :, 0:3])
            points_DA = torch.Tensor(points_DA)
            points_DA = points_DA.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()
                points_DA = points_DA.cuda()

            pred, trans_feat = classifier(points)

            # Multi-layer Loss
            ###############################################################################################
            # FC1
            classifier.fc1.register_forward_hook(get_activation('fc1'))
            output_dense_1 = classifier(points)
            feature_dense_1 = activation['fc1']
            # print(feature_dense_1.size())

            classifier.fc1.register_forward_hook(get_activation('fc1'))
            output_DA_1 = classifier(points_DA)
            feature_DA_1 = activation['fc1']
            # print(feature_DA_1.size())

            # FC2
            classifier.fc2.register_forward_hook(get_activation('fc2'))
            output_dense_2 = classifier(points)
            feature_dense_2 = activation['fc2']
            # print(feature_dense_2.size())

            classifier.fc2.register_forward_hook(get_activation('fc2'))
            output_DA_2 = classifier(points_DA)
            feature_DA_2 = activation['fc2']
            # print(feature_DA_2.size())

            # change the loss here for testing!!!

            DA_loss, loss = criterion_DA(pred, target.long(), trans_feat,
                                feature_dense_1, feature_DA_1, feature_dense_2, feature_DA_2)
            ################################################################################################
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

            loss.backward()
            optimizer.step()
            global_step += 1

            # Print the loss
            running_loss += DA_loss.item()
            if batch_id % 100 == 99:
                # log_string("fc1 {}".format(classifier.fc1.weight.grad))
                # log_string("fc2 {}".format(classifier.fc2.weight.grad))
                # log_string("fc3 {}".format(classifier.fc3.weight.grad))
                # print("Training loss {} ".format(loss.item()/100))
                calculate_loss = running_loss/100
                log_string("Running DA loss {} ".format(calculate_loss))

                if calculate_loss < min_loss:
                    min_loss = calculate_loss
                    logger.info('Save model...')
                    savepath = str(checkpoints_dir) + '/best_model.pth'
                    log_string('Saving at %s' % savepath)
                    state = {
                        'epoch': epoch,
                        # 'instance_acc': instance_acc,
                        # 'class_acc': class_acc,
                        'loss': calculate_loss,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
                running_loss = 0.0

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                # logger.info('Save model...')
                # # print("This is a better model, but the model will not be saved")
                logger.info('Model will not be saved with vision validation')
                # savepath = str(checkpoints_dir) + '/best_model.pth'
                # log_string('Saving at %s' % savepath)
                # state = {
                #     'epoch': best_epoch,
                #     'instance_acc': instance_acc,
                #     'class_acc': class_acc,
                #     'model_state_dict': classifier.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                # }
                # torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    # torch.cuda.empty_cache()
    args = parse_args()
    main(args)
