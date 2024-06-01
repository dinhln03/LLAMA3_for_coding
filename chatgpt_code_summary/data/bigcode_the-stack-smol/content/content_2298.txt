import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pickle
from data_load import mnist, svhn, usps

# inverse_transform = None

# class InverseTransform(torchvision.transforms.Normalize):
#     """
#     Undoes the normalization and returns the reconstructed images in the input domain.
#     """

#     def __init__(self, mean, std):
#         mean = torch.as_tensor(mean)
#         std = torch.as_tensor(std)
#         std_inv = 1 / (std + 1e-7)
#         mean_inv = -mean * std_inv
#         super().__init__(mean=mean_inv, std=std_inv)

#     def __call__(self, tensor):
#         t = super().__call__(tensor.clone())
#         # return transforms.ToPILImage()(t)
#         return t


def digit_load(args): 
    global inverse_transform
    train_bs = args.batch_size
    if args.dset == 's':
        test_source = svhn.SVHN('./data/svhn/', split='test', download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))  
        # assert inverse_transform == None
        # inverse_transform = InverseTransform((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    elif args.dset == 'u':
        test_source = usps.USPS('./data/usps/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(28, padding=4),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))    
        # assert inverse_transform == None
        # inverse_transform = InverseTransform((0.5,), (0.5,))
    elif args.dset == 'm':
        test_source = mnist.MNIST('./data/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
        # assert inverse_transform == None
        # inverse_transform = InverseTransform((0.5,), (0.5,))

    dset_loaders = {}
    dset_loaders["test"] = DataLoader(test_source, batch_size=train_bs*2, shuffle=False, 
        num_workers=args.worker, drop_last=False)
    
    return dset_loaders

def cal_acc(loader, netF, netB, netC):
    k = 0
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            input_images = []
            inputs = data[0]
            inputs_clone = inputs.clone()
            for j in range(inputs_clone.size(0)):
                x = transforms.Normalize((-1,), (2,))(inputs_clone[j])
                input_images.append(transforms.ToPILImage()(x))
            labels = data[1]
            outputs = netC(netB(netF(inputs)))
            #
            _, predict = torch.max(outputs.float().cpu(), 1)
            for j in range(inputs.size(0)):
                folder = args.output_dir + '/inspect/label-{}'.format(labels[j])
                if not osp.exists(folder):
                    os.makedirs(folder)
                subfolder = folder + '/pred-{}'.format(predict[j])
                if not osp.exists(subfolder):
                    os.makedirs(subfolder)
                input_images[j].save(subfolder + '/{}.jpg'.format(k))
                k += 1
            #
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy*100, mean_ent

def test(args):
    dset_loaders = digit_load(args)
    ## set base network
    if args.dset == 'u':
        netF = network.LeNetBase()#.cuda()
    elif args.dset == 'm':
        netF = network.LeNetBase()#.cuda()  
    elif args.dset == 's':
        netF = network.DTNBase()#.cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck)#.cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck)#.cuda()

    args.modelpath = args.output_dir + '/F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/B.pt'   
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/C.pt'   
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC)
    log_str = 'Task: {}, Accuracy = {:.2f}%'.format(args.dset, acc)
    try: 
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
    except:
        pass
    print(log_str+'\n')

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=30, help="maximum epoch")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='s', choices=['u', 'm','s'])
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()
    args.class_num = 10

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    args.output_dir = osp.join(args.output, 'seed' + str(args.seed), args.dset)

    test(args)

# python unsupervised_digit.py --dset m --gpu_id 0 --output ckps_unsupervised_digit
# python unsupervised_digit.py --dset m --gpu_id 0 --ent --output ckps_unsupervised_digit_ent
# python unsupervised_digit.py --dset m --gpu_id 0 --gent --output ckps_unsupervised_digit_gent
# python unsupervised_digit.py --dset m --gpu_id 0 --ent --gent --output ckps_unsupervised_digit_ent_gent

# na verdade n sem como saber qual classe vai sair .. ideal é ver tsne? ou mostrar as classificacoes primeiro?
# show classification + gradcam (versao mais rapida)