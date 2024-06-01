import argparse
import torch
import os
import numpy as np
import random as rd

from models import GCN
from utils import get_folder_path
from base_solver import BaseSolver


MODEL = 'GCN'

parser = argparse.ArgumentParser()
# Dataset params
parser.add_argument("--dataset", type=str, default='Movielens', help="")
parser.add_argument("--dataset_name", type=str, default='1m', help="")
parser.add_argument("--if_use_features", type=bool, default=False, help="")
parser.add_argument("--num_core", type=int, default=10, help="")
parser.add_argument("--num_feat_core", type=int, default=10, help="")
parser.add_argument("--train_ratio", type=float, default=0.8, help="")
# Model params
parser.add_argument("--dropout", type=float, default=0.5, help="")
parser.add_argument("--emb_dim", type=int, default=64, help="")
parser.add_argument("--repr_dim", type=int, default=16, help="")
parser.add_argument("--hidden_size", type=int, default=64, help="")
# Train params
parser.add_argument("--num_negative_samples", type=int, default=5, help="")
parser.add_argument("--init_eval", type=bool, default=True, help="")

parser.add_argument("--device", type=str, default='cuda', help="")
parser.add_argument("--gpu_idx", type=str, default='0', help="")
parser.add_argument("--runs", type=int, default=100, help="")
parser.add_argument("--epochs", type=int, default=100, help="")
parser.add_argument("--opt", type=str, default='adam', help="")
parser.add_argument("--loss", type=str, default='mse', help="")
parser.add_argument("--batch_size", type=int, default=4, help="")
parser.add_argument("--lr", type=float, default=1e-4, help="")
parser.add_argument("--weight_decay", type=float, default=1e-3, help="")
parser.add_argument("--early_stopping", type=int, default=60, help="")
parser.add_argument("--save_epochs", type=list, default=[10, 40, 80], help="")
parser.add_argument("--save_every_epoch", type=int, default=40, help="")

args = parser.parse_args()


# Setup data and weights file path
data_folder, weights_folder, logger_folder = \
    get_folder_path(model=MODEL, dataset=args.dataset + args.dataset_name)

# Setup device
if not torch.cuda.is_available() or args.device == 'cpu':
    device = 'cpu'
else:
    device = 'cuda:{}'.format(args.gpu_idx)

# Setup args
dataset_args = {
    'root': data_folder, 'dataset': args.dataset, 'name': args.dataset_name,
    'if_use_features': args.if_use_features,
    'num_core': args.num_core, 'num_feat_core': args.num_feat_core,
    'train_ratio': args.train_ratio
}
model_args = {
    'if_use_features': args.if_use_features,
    'emb_dim': args.emb_dim, 'hidden_size': args.hidden_size,
    'repr_dim': args.repr_dim, 'dropout': args.dropout
}
train_args = {
    'init_eval': args.init_eval, 'num_negative_samples': args.num_negative_samples,
    'opt': args.opt, 'loss': args.loss,
    'runs': args.runs, 'epochs': args.epochs, 'batch_size': args.batch_size,
    'weight_decay': args.weight_decay, 'lr': args.lr, 'device': device,
    'weights_folder': os.path.join(weights_folder, str(model_args)),
    'logger_folder': os.path.join(logger_folder, str(model_args)),
    'save_epochs': args.save_epochs, 'save_every_epoch': args.save_every_epoch
}
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(model_args))
print('train params: {}'.format(train_args))


class GCNSolver(BaseSolver):
    def __init__(self, GCN, dataset_args, model_args, train_args):
        super(GCNSolver, self).__init__(GCN, dataset_args, model_args, train_args)

    def prepare_model_input(self, data, if_use_features=False):
        edge_index_np = np.hstack(data.edge_index_nps[0].values())
        edge_index_np = np.hstack([edge_index_np, np.flip(edge_index_np, 0)])
        edge_index = torch.from_numpy(edge_index_np).long().to(self.train_args['device'])

        kwargs = {'edge_index': edge_index}
        if if_use_features:
            kwargs['x'] = data.x

        return kwargs

    def train_negative_sampling(self, u_nid, train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map, data):
        """
        Unliked popular movie negative sampling:
        :param u_nid:
        :param train_pos_unid_inid_map:
        :param test_pos_unid_inid_map:
        :param neg_unid_inid_map:
        :param data:
        :return:
        """
        num_pos_samples = len(train_pos_unid_inid_map[u_nid])

        negative_inids = test_pos_unid_inid_map[u_nid] + neg_unid_inid_map[u_nid]
        nid_occs = np.array([data.item_nid_occs[0][nid] for nid in negative_inids])
        nid_occs = nid_occs / np.sum(nid_occs)
        negative_inids = rd.choices(population=negative_inids, weights=nid_occs, k=num_pos_samples * 5)

        return negative_inids

    def generate_candidates(self, train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map, u_nid):
        pos_i_nids = test_pos_unid_inid_map[u_nid]
        neg_i_nids = np.array(neg_unid_inid_map[u_nid])

        neg_i_nids_indices = np.array(rd.sample(range(neg_i_nids.shape[0]), 99), dtype=int)

        return pos_i_nids, list(neg_i_nids[neg_i_nids_indices])


if __name__ == '__main__':
    solver = GCNSolver(GCN, dataset_args, model_args, train_args)
    solver.run()
