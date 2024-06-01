import argparse
import os
import os.path as osp
import shutil
import tempfile
import json
import pdb
import numpy as np
import pickle
import pandas as pd
import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint

from mmdet.apis import init_dist
from mmdet.core import lvis_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.core import build_assigner
from utils import filter_logits_by_gt

TEMP_DATASET_SIZE = 5000

def single_gpu_test(model, data_loader, show=False, cfg=None, index=0, img_meta=None):
    model.eval()
    results = []
    logits_list = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    class_instances = pickle.load(open('train_instances_list.p', 'rb'))
    normalized_classes = np.zeros(1231)
    for i, c in enumerate(class_instances):
        if c:
            normalized_classes[i] = 1/np.sqrt(c)
    for i, data in enumerate(data_loader):
    #     if i < TEMP_DATASET_SIZE*index:
    #         continue
        if i >= TEMP_DATASET_SIZE*(index+1):   # temporary condition for testing
            break
        with torch.no_grad():
            bbox_results, det_bboxes, det_labels, scores = model(return_loss=False, rescale=not show, **data, img_id=i, norm_cls=normalized_classes)
            det_bboxes = det_bboxes.detach().cpu()
            det_labels = det_labels.detach().cpu()
            scores = scores.detach().cpu()
        # save original logits:
        # filename = data['img_meta'][0].data[0][0]['filename'].split('/')[-1]  # get the file name, e.g: '000000397133.jpg'
        # with open(f'test_logits/logits_per_img/{filename}.p', 'wb') as outfile:
        #     pickle.dump(scores, outfile)
        results.append(bbox_results)
        logits_list.append((det_bboxes, det_labels, scores))

        if show:
            model.module.show_result(data, bbox_results)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results, logits_list  # return also class. logits and labels

def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--tau', type=float, default=0.0)
    parser.add_argument('--data_index', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def reweight_cls(model, tauuu):

    if tauuu == 0:
        return model

    model_dict = model.state_dict()

    def pnorm(weights, tau):
        normB = torch.norm(weights, 2, 1)
        ws = weights.clone()

        for i in range(0, weights.shape[0]):
            ws[i] = ws[i] / torch.pow(normB[i], tau)

        return ws

    reweight_set = ['bbox_head.fc_cls.weight']
    tau = tauuu
    for k in reweight_set:
        weight = model_dict[k]  # ([1231, 1024])
        weight = pnorm(weight, tau)
        model_dict[k].copy_(weight)
        print('Reweight param {:<30} with tau={}'.format(k, tau))

    return model


def logits_process(logits):
    """
    Get the logits as a tuple of softmax logits ,bounding boxes and labels.
    Output: to matrices:
    logits_mat in size (dataset, 300, 1231) - top 300 logits for each image.
    bboxes_mat in size (dataset, 300, 4) - top 300 bboxes for each image.
    labels_mat in size (dataset, 300, 1) - corresponding labels. 300 for each image.
    """
    # all_bboxes_logits = []
    # for image in logits:
    #     image_bboxes_logits = []
    #     for i, bbox in enumerate(image[0]):
    #         bboxes_logits_dict = dict()  # image[0] = tensor including 300 bboxes
    #         index = int(bbox[5].item())  # bbox[6] specifies the relevant line in the logits matrix
    #         logits_vector = image[1][index]
    #         bboxes_logits_dict['bbox'] = bbox[:4]
    #         bboxes_logits_dict['score'] = bbox[4]
    #         bboxes_logits_dict['logits'] = logits_vector
    #         image_bboxes_logits.append(bboxes_logits_dict)
    #     all_bboxes_logits.append(image_bboxes_logits)


    # for idx in range(len(dataset)):
    #     img_id = dataset.img_ids[idx]

    logits_mat = np.zeros((TEMP_DATASET_SIZE, 300, 1231))
    bboxes_mat = np.zeros((TEMP_DATASET_SIZE, 300, 4))
    labels_mat = np.zeros((TEMP_DATASET_SIZE, 300))
    proposal_num = np.zeros((TEMP_DATASET_SIZE, 300, 1))
    for i, image in enumerate(logits):
        for j, bbox in enumerate(image[0]):  # image[0] = tensor including 300 bboxes
            # bboxes_logits_dict = dict()
            index = int(bbox[5].item())  # bbox[5] specifies the relevant line in the logits matrix
            logits_vector = image[2][index]  # image[2] includes the scores
            # bbox_arr = np.array(bbox[:4])
            bboxes_mat[i][j][:] = bbox[:4]
            logits_mat[i][j] = np.array(logits_vector)
            # added this to compute proposal numbers
            proposal_num[i][j] = bbox[-1]
        labels_mat[i] = image[1]  # image[1] includes the labels

    return bboxes_mat, labels_mat, logits_mat, proposal_num


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.data_index % 2)
    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)  # original - test | changed to test_with_train_data
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=0,  # cfg.data.workers_per_gpu
        dist=distributed,
        shuffle=False)

    # save gt boxes and labels for learning nms
    # for i, data in enumerate(data_loader):
    #     img_id = dataset.img_infos[i]['id']
    #     gt = dataset.get_ann_info(i)
    #     gt_boxes = gt['bboxes']
    #     gt_labels = gt['labels']
    #     filename = f'test_logits/learning_nms_data/{i}/gt_boxes.p'  # file name for new directory
    #     os.makedirs(os.path.dirname(filename), exist_ok=True)
    #     with open(f'test_logits/learning_nms_data/{i}/gt_boxes.p', 'wb') as outfile:  # possible to include img_id
    #         pickle.dump(gt_boxes, outfile)
    #     with open(f'test_logits/learning_nms_data/{i}/gt_labels.p', 'wb') as outfile:
    #         pickle.dump(gt_boxes, outfile)
    #
    #     # filename = dataset.img_infos[i]['filename']
    #     # with open(f'test_gt/{filename}.p', 'wb') as outfile:
    #     #     pickle.dump(gt_labels, outfile)

    # save gt instances per class
    # instances_list = np.zeros(1231)
    # for i, data in enumerate(data_loader):  # original script in test_lvis_tnorm.py
    #     gt = dataset.get_ann_info(i)
    #     print(i)
    #     for label in gt['labels']:
    #         instances_list[label] += 1
    # with open('train_instances_list.p', 'wb') as outfile:
    #     pickle.dump(instances_list, outfile)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = reweight_cls(model, args.tau)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs, logits = single_gpu_test(model, data_loader, args.show, cfg, args.data_index)
    else:
        model = MMDistributedDataParallel(model.cuda())
        outputs = multi_gpu_test(model, data_loader, args.tmpdir)

    # save outputs as csv:
    # pd.DataFrame(outputs).to_csv("original_outputs_full.csv")
    # preprocess logits and save them on json file
    # otp = np.asarray(outputs)  # temp
    # df = pd.DataFrame(otp)
    # df.to_csv('otp.csv', index=False)

    bboxes_mat, labels_mat, logits_mat, proposal_num = logits_process(logits)

    # save labels, boxes and logits
    # with open('test_logits/dragon_test_bboxes_mat.p', 'wb') as outfile:
    #     pickle.dump(bboxes_mat, outfile)
    # with open('test_logits/dragon_labels_mat.p', 'wb') as outfile:
    #     pickle.dump(labels_mat, outfile)
    # with open('logits_mat1.p', 'wb') as outfile:
    #     pickle.dump(logits_mat[:1000], outfile)
    # with open('logits_mat2.p', 'wb') as outfile:
    #     pickle.dump(logits_mat[1000:2000], outfile)
    # with open('logits_mat3.p', 'wb') as outfile:
    #     pickle.dump(logits_mat[2000:3000], outfile)
    # with open('logits_mat4.p', 'wb') as outfile:
    #     pickle.dump(logits_mat[3000:4000], outfile)
    # with open('logits_mat5.p', 'wb') as outfile:
    #     pickle.dump(logits_mat[4000:], outfile)

    # filter detections by iou with gt (for dragon training)
    gt_list = []
    results_per_image = []
    for i, data in enumerate(data_loader):  # original script in test_lvis_tnorm.py
        # if i < TEMP_DATASET_SIZE*args.data_index:
        #     continue
        if i >= TEMP_DATASET_SIZE:   # temporary condition for testing
            break
        print(i)
        img_id = dataset.img_infos[i]['id']
        gt = dataset.get_ann_info(i)
        gt_dict = dict()
        gt_dict['id'] = img_id
        gt_dict['bboxes'] = gt['bboxes']
        gt_dict['labels'] = gt['labels']
        gt_list.append(gt_dict)
        # filter logits according to equivalent ground truth.
        # after filtering, for each image we get a list in length of classes and detections belongs to this class.
        results = filter_logits_by_gt(bboxes_mat[i], logits_mat[i], gt_list[i], proposal_num[i], i)
        results_per_image.append(results)
    with open(f'dragon_bboxes_logits_map24.p', 'wb') as outfile:
        pickle.dump(results_per_image, outfile)
    print('saved')

    # evaluation:
    rank, _ = get_dist_info()
    if args.out and rank == 0:
        print('\nwriting results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)
        eval_types = args.eval
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if eval_types == ['proposal_fast']:
                result_file = args.out
                lvis_eval(result_file, eval_types, dataset.lvis)
            else:
                if not isinstance(outputs[0], dict):
                    result_files = results2json(dataset, outputs, args.out, args.data_index)
                    lvis_eval(result_files, eval_types, dataset.lvis, max_dets=300)
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = args.out + '.{}'.format(name)
                        result_files = results2json(dataset, outputs_,
                                                    result_file)
                        lvis_eval(result_files, eval_types, dataset.lvis)



    # Save predictions in the COCO json format
    if args.json_out and rank == 0:
        if not isinstance(outputs[0], dict):
            results2json(dataset, outputs, args.json_out)
        else:
            for name in outputs[0]:
                outputs_ = [out[name] for out in outputs]
                result_file = args.json_out + '.{}'.format(name)
                results2json(dataset, outputs_, result_file)


if __name__ == '__main__':
    main()
