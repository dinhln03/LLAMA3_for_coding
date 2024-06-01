import sys
import cv2
import os
from ast import literal_eval
from pathlib import Path
import shutil
import logging
import random
import pickle
import yaml
import subprocess
from PIL import Image
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation, rc
plt.rcParams['figure.figsize'] = 30, 30
np.set_printoptions(precision=3, suppress=True)
rc('animation', html='jshtml')

import torch

from augmentations import get_albu_transforms


IMAGE_DIR = '~/Kaggle/data/tensorflow-great-barrier-reef/train_images'

def load_image(video_id, video_frame, image_dir):
    img_path = f'{image_dir}/video_{video_id}/{video_frame}.jpg'
    assert os.path.exists(img_path), f'{img_path} does not exist.'
    img = cv2.imread(img_path)
    return img

def decode_annotations(annotaitons_str):
    """decode annotations in string to list of dict"""
    return literal_eval(annotaitons_str)


def load_image_with_annotations(video_id, video_frame, image_dir, annotaitons_str):
    img = load_image(video_id, video_frame, image_dir)
    annotations = decode_annotations(annotaitons_str)
    if len(annotations) > 0:
        for ann in annotations:
            cv2.rectangle(img, (ann['x'], ann['y']),
                (ann['x'] + ann['width'], ann['y'] + ann['height']),
                (255, 0, 0), thickness=2,)
    return img

def draw_predictions(img, pred_bboxes):
    img = img.copy()
    if len(pred_bboxes) > 0:
        for bbox in pred_bboxes:
            conf = bbox[0]
            x, y, w, h = bbox[1:].round().astype(int)
            cv2.rectangle(img, (x, y),(x+w, y+h),(0, 255, 255), thickness=2,)
            cv2.putText(img, f"{conf:.2}",(x, max(0, y-5)),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),
                thickness=1,
            )
    return img

def plot_img(df, idx, image_dir, pred_bboxes=None):
    row = df.iloc[idx]
    video_id = row.video_id
    video_frame = row.video_frame
    annotations_str = row.annotations
    img = load_image_with_annotations(video_id, video_frame, image_dir, annotations_str)
    
    if pred_bboxes and len(pred_bboxes) > 0:
        pred_bboxes = pred_bboxes[pred_bboxes[:,0].argsort()[::-1]] # sort by conf
        img = draw_predictions(img, pred_bboxes)
    plt.imshow(img[:, :, ::-1])    
    
def calc_iou(bboxes1, bboxes2, bbox_mode='xywh'):
    assert len(bboxes1.shape) == 2 and bboxes1.shape[1] == 4
    assert len(bboxes2.shape) == 2 and bboxes2.shape[1] == 4
    
    bboxes1 = bboxes1.copy()
    bboxes2 = bboxes2.copy()
    
    if bbox_mode == 'xywh':
        bboxes1[:, 2:] += bboxes1[:, :2]
        bboxes2[:, 2:] += bboxes2[:, :2]

    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1e-9), 0) * np.maximum((yB - yA + 1e-9), 0)
    boxAArea = (x12 - x11 + 1e-9) * (y12 - y11 + 1e-9)
    boxBArea = (x22 - x21 + 1e-9) * (y22 - y21 + 1e-9)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou

def f_beta(tp, fp, fn, beta=2):
    if tp == 0:
        return 0
    return (1+beta**2)*tp / ((1+beta**2)*tp + beta**2*fn+fp)

def calc_is_correct_at_iou_th(gt_bboxes, pred_bboxes, iou_th, verbose=False):
    gt_bboxes = gt_bboxes.copy()
    pred_bboxes = pred_bboxes.copy()
    
    tp = 0
    fp = 0
    for k, pred_bbox in enumerate(pred_bboxes): # fixed in ver.7
        if len(gt_bboxes) == 0:
            fp += len(pred_bboxes) - k # fix in ver.7
            break
        ious = calc_iou(gt_bboxes, pred_bbox[None, 1:])
        max_iou = ious.max()
        if max_iou >= iou_th:
            tp += 1
            gt_bboxes = np.delete(gt_bboxes, ious.argmax(), axis=0)
        else:
            fp += 1

    fn = len(gt_bboxes)
    return tp, fp, fn

def calc_is_correct(gt_bboxes, pred_bboxes, iou_th=0.5):
    """
    gt_bboxes: (N, 4) np.array in xywh format
    pred_bboxes: (N, 5) np.array in conf+xywh format
    """
    if len(gt_bboxes) == 0 and len(pred_bboxes) == 0:
        tps, fps, fns = 0, 0, 0
        return tps, fps, fns

    elif len(gt_bboxes) == 0:
        tps, fps, fns = 0, len(pred_bboxes), 0
        return tps, fps, fns

    elif len(pred_bboxes) == 0:
        tps, fps, fns = 0, 0, len(gt_bboxes)
        return tps, fps, fns

    pred_bboxes = pred_bboxes[pred_bboxes[:,0].argsort()[::-1]] # sort by conf

    tps, fps, fns = 0, 0, 0
    tp, fp, fn = calc_is_correct_at_iou_th(gt_bboxes, pred_bboxes, iou_th)
    tps += tp
    fps += fp
    fns += fn
    return tps, fps, fns

def calc_f2_score(gt_bboxes_list, pred_bboxes_list, verbose=False):
    """
    gt_bboxes_list: list of (N, 4) np.array in xywh format
    pred_bboxes_list: list of (N, 5) np.array in conf+xywh format
    """
    #f2s = []
    f2_dict = {'f2':0, "P":0, "R": 0}
    all_tps = [list([0] * 11) for _  in range(len(gt_bboxes_list))]
    all_fps = [list([0] * 11) for _  in range(len(gt_bboxes_list))]
    all_fns = [list([0] * 11) for _  in range(len(gt_bboxes_list))]
    for k, iou_th in enumerate(np.arange(0.3, 0.85, 0.05)):
        tps, fps, fns = 0, 0, 0
        for i, (gt_bboxes, pred_bboxes) in enumerate(zip(gt_bboxes_list, pred_bboxes_list)):
            tp, fp, fn = calc_is_correct(gt_bboxes, pred_bboxes, iou_th)
            tps += tp
            fps += fp
            fns += fn
            all_tps[i][k] = tp
            all_fps[i][k] = fp
            all_fns[i][k] = fn
            if verbose:
                num_gt = len(gt_bboxes)
                num_pred = len(pred_bboxes)
                print(f'num_gt:{num_gt:<3} num_pred:{num_pred:<3} tp:{tp:<3} fp:{fp:<3} fn:{fn:<3}')
        f2 = f_beta(tps, fps, fns, beta=2)    
        precision = f_beta(tps, fps, fns, beta=0)
        recall = f_beta(tps, fps, fns, beta=100)
        f2_dict["f2_" + str(round(iou_th,3))] = f2
        f2_dict["P_" + str(round(iou_th,3))] = precision
        f2_dict["R_" + str(round(iou_th,3))] = recall
        f2_dict['f2'] += f2 / 11
        f2_dict['P'] += precision / 11
        f2_dict['R'] += recall / 11
    f2_dict["tps"] = all_tps
    f2_dict["fps"] = all_fps
    f2_dict["fns"] = all_fns
    return f2_dict

def print_f2_dict(d):
    print("Overall f2: {:.3f}, precision {:.3f}, recall {:.3f}".format(d['f2'], d['precision'], d['recall']))
    for k, iou_th in enumerate(np.arange(0.3, 0.85, 0.05)):
        print(f"IOU {iou_th:.2f}:", end=" ")
        print("f2: {:.3f}, precision {:.3f}, recall {:.3f}".format(d["f2_" + str(round(iou_th,3))], 
                                                                   d["precision_" + str(round(iou_th,3))], 
                                                                   d["recall_" + str(round(iou_th,3))]))
        
    

def get_path(row, params, infer=False):
    row['old_image_path'] = params['root_dir'] / f'train_images/video_{row.video_id}/{row.video_frame}.jpg'
    if infer:
        row['image_path'] = row["old_image_path"]
    else:
        row['image_path'] = params['image_dir'] / f'video_{row.video_id}_{row.video_frame}.jpg'
    row['label_path'] = params['label_dir'] / f'video_{row.video_id}_{row.video_frame}.txt'
    return row

def make_copy(path, params):
    # TODO: fix split issue
    data = str(path).split('/')
    filename = data[-1]
    video_id = data[-2]
    new_path = params["image_dir"] / f'{video_id}_{filename}'
    shutil.copy(path, new_path)
    return

# https://www.kaggle.com/awsaf49/great-barrier-reef-yolov5-train
def voc2yolo(image_height, image_width, bboxes):
    """
    voc  => [x1, y1, x2, y1]
    yolo => [xmid, ymid, w, h] (normalized)
    """
    
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]]/ image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]]/ image_height
    
    w = bboxes[..., 2] - bboxes[..., 0]
    h = bboxes[..., 3] - bboxes[..., 1]
    
    bboxes[..., 0] = bboxes[..., 0] + w/2
    bboxes[..., 1] = bboxes[..., 1] + h/2
    bboxes[..., 2] = w
    bboxes[..., 3] = h
    
    return bboxes

def yolo2voc(image_height, image_width, bboxes):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    voc  => [x1, y1, x2, y1]
    
    """ 
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]]* image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]]* image_height
    
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]]/2
    bboxes[..., [2, 3]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]
    
    return bboxes

def coco2yolo(image_height, image_width, bboxes):
    """
    coco => [xmin, ymin, w, h]
    yolo => [xmid, ymid, w, h] (normalized)
    """
    
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    # normolizinig
    bboxes[..., [0, 2]]= bboxes[..., [0, 2]]/ image_width
    bboxes[..., [1, 3]]= bboxes[..., [1, 3]]/ image_height
    
    # converstion (xmin, ymin) => (xmid, ymid)
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]/2
    
    return bboxes

def yolo2coco(image_height, image_width, bboxes):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    coco => [xmin, ymin, w, h]
    
    """ 
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    # denormalizing
    bboxes[..., [0, 2]]= bboxes[..., [0, 2]]* image_width
    bboxes[..., [1, 3]]= bboxes[..., [1, 3]]* image_height
    
    # converstion (xmid, ymid) => (xmin, ymin) 
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]]/2
    
    return bboxes

def voc2coco(bboxes, image_height=720, image_width=1280):
    bboxes  = voc2yolo(image_height, image_width, bboxes)
    bboxes  = yolo2coco(image_height, image_width, bboxes)
    return bboxes


def load_image(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        
        
def draw_bboxes(img, bboxes, classes, colors = None, show_classes = None, bbox_format = 'yolo', class_name = False, line_thickness = 1):  
     
    image = img.copy()
    show_classes = classes if show_classes is None else show_classes
    colors = (0, 255 ,0) if colors is None else colors
    
    if bbox_format == 'yolo':
        
        for idx in range(len(bboxes)):  
            
            bbox  = bboxes[idx]
            cls   = classes[idx]
            color = colors[idx]
            if cls in show_classes:
            
                x1 = round(float(bbox[0])*image.shape[1])
                y1 = round(float(bbox[1])*image.shape[0])
                w  = round(float(bbox[2])*image.shape[1]/2) #w/2 
                h  = round(float(bbox[3])*image.shape[0]/2)

                voc_bbox = (x1-w, y1-h, x1+w, y1+h)
                plot_one_box(voc_bbox, 
                             image,
                             color = color,
                             label = cls if class_name else str(get_label(cls)),
                             line_thickness = line_thickness)
            
    elif bbox_format == 'coco':
        
        for idx in range(len(bboxes)):  
            
            bbox  = bboxes[idx]
            cls   = classes[idx]
            color = colors[idx]
            
            if cls in show_classes:            
                x1 = int(round(bbox[0]))
                y1 = int(round(bbox[1]))
                w  = int(round(bbox[2]))
                h  = int(round(bbox[3]))

                voc_bbox = (x1, y1, x1+w, y1+h)
                plot_one_box(voc_bbox, 
                             image,
                             color = color,
                             label = cls,
                             line_thickness = line_thickness)

    elif bbox_format == 'voc_pascal':
        
        for idx in range(len(bboxes)):  
            
            bbox  = bboxes[idx]
            cls   = classes[idx]
            cls_id = class_ids[idx]
            color = colors[cls_id] if type(colors) is list else colors
            
            if cls in show_classes: 
                x1 = int(round(bbox[0]))
                y1 = int(round(bbox[1]))
                x2 = int(round(bbox[2]))
                y2 = int(round(bbox[3]))
                voc_bbox = (x1, y1, x2, y2)
                plot_one_box(voc_bbox, 
                             image,
                             color = color,
                             label = cls if class_name else str(cls_id),
                             line_thickness = line_thickness)
    else:
        raise ValueError('wrong bbox format')

    return image

def get_bbox(annots):
    bboxes = [list(annot.values()) for annot in annots]
    return bboxes

def get_imgsize(row):
    row['width'], row['height'] = imagesize.get(row['image_path'])
    return row


# https://www.kaggle.com/diegoalejogm/great-barrier-reefs-eda-with-animations
def create_animation(ims):
    fig = plt.figure(figsize=(16, 12))
    plt.axis('off')
    im = plt.imshow(ims[0])

    def animate_func(i):
        im.set_array(ims[i])
        return [im]

    return animation.FuncAnimation(fig, animate_func, frames = len(ims), interval = 1000//12)

# https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.py
def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

# https://github.com/DocF/Soft-NMS/blob/master/soft_nms.py

def py_cpu_softnms(dets, sc, Nt=0.3, sigma=0.5, thresh=0.001, method=2):
    """
    py_cpu_softnms
    :param dets:   boexs 坐标矩阵 format [y1, x1, y2, x2]
    :param sc:     每个 boxes 对应的分数
    :param Nt:     iou 交叠门限
    :param sigma:  使用 gaussian 函数的方差
    :param thresh: 最后的分数门限
    :param method: 使用的方法
    :return:       留下的 boxes 的 index
    """

    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # the order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = sc
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        #
        if i != N-1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # IoU calculate
        xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
        yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
        xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
        yy2 = np.minimum(dets[i, 2], dets[pos:, 2])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    inds = dets[:, 4][scores > thresh]
    keep = inds.astype(int)

    return keep


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    
def create_logger(filename, filemode='a'):
    # better logging file - output the in terminal as well
    file_handler = logging.FileHandler(filename=filename, mode=filemode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    formatter = "%(asctime)s %(levelname)s: %(message)s"
    datefmt = "%m/%d/%Y %I:%M:%S %p"
    logging.basicConfig(format=formatter, datefmt=datefmt, 
                        level=logging.DEBUG, handlers=handlers)
    return

def save_pickle(obj, folder_path):
    pickle.dump(obj, open(folder_path, 'wb'), pickle.HIGHEST_PROTOCOL)

def load_pickle(folder_path):
    return pickle.load(open(folder_path, 'rb'))

def save_yaml(obj, folder_path):
    obj2 = obj.copy()
    for key, value in obj2.items():
        if isinstance(value, Path):
            obj2[key] = str(value.resolve())
        else:
            obj2[key] = value
    with open(folder_path, 'w') as file:
        yaml.dump(obj2, file)    
        
def load_yaml(folder_path):
    with open(folder_path) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data

def load_model(params):
    try:
        model = torch.hub.load(params['repo'],
                            'custom',
                            path=params['ckpt_path'],
                            source='local',
                            force_reload=True)  # local repo
    except:
        print("torch.hub.load failed, try torch.load")
        model = torch.load(params['ckpt_path'])
    model.conf = params['conf']  # NMS confidence threshold
    model.iou  = params['iou']  # NMS IoU threshold
    model.classes = None   # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 50  # maximum number of detections per image
    return model


def predict(model, img, size=768, augment=False, use_sahi=False):
    if use_sahi:
        from sahi.predict import get_sliced_prediction
        results = get_sliced_prediction(
            img,
            model,
            slice_height = 512,
            slice_width = 512,
            overlap_height_ratio = 0.2,
            overlap_width_ratio = 0.2
        ) 
        preds = results.object_prediction_list
        bboxes = np.array([pred.bbox.to_voc_bbox() for pred in preds])                
    else:        
        results = model(img, size=size, augment=augment)  # custom inference size
        preds   = results.pandas().xyxy[0]
        bboxes  = preds[['xmin','ymin','xmax','ymax']].values
    if len(bboxes):
        height, width = img.shape[:2]
        bboxes  = voc2coco(bboxes,height,width).astype(int)
        if use_sahi:
            confs   = np.array([pred.score.value for pred in preds])
        else:
            confs   = preds.confidence.values
        return bboxes, confs
    else:
        return np.array([]),[]
    
def format_prediction(bboxes, confs):
    annot = ''
    if len(bboxes)>0:
        for idx in range(len(bboxes)):
            xmin, ymin, w, h = bboxes[idx]
            conf = confs[idx]
            annot += f'{conf} {xmin} {ymin} {w} {h}'
            annot +=' '
        annot = annot.strip(' ')
    return annot

def show_img(img, bboxes, confs, colors, bbox_format='yolo'):
    labels = [str(round(conf,2)) for conf in confs]
    img    = draw_bboxes(img = img,
                           bboxes = bboxes, 
                           classes = labels,
                           class_name = True, 
                           colors = colors, 
                           bbox_format = bbox_format,
                           line_thickness = 2)

    return Image.fromarray(img)


def write_hyp(params):    
    with open(params["hyp_file"], mode="w") as f:
        for key, val in params["hyp_param"].items():
            f.write(f"{key}: {val}\n")
        
def class2dict(f):
    return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))


def upload(params):
    data_version = "-".join(params["exp_name"].split("_"))
    if os.path.exists(params["output_dir"] / "wandb"):
        shutil.move(str(params["output_dir"] / "wandb"), 
                    str(params["output_dir"].parent / f"{params['exp_name']}_wandb/")
        )        
    with open(params["output_dir"] / "dataset-metadata.json", "w") as f:
        f.write("{\n")
        f.write(f"""  "title": "{data_version}",\n""")
        f.write(f"""  "id": "vincentwang25/{data_version}",\n""")
        f.write("""  "licenses": [\n""")
        f.write("""    {\n""")
        f.write("""      "name": "CC0-1.0"\n""")
        f.write("""    }\n""")
        f.write("""  ]\n""")
        f.write("""}""")    
    subprocess.call(["kaggle", "datasets", "create", "-p", str(params["output_dir"]), "-r", "zip"])
    
def coco(df):
    annotion_id = 0
    images = []
    annotations = []

    categories = [{'id': 0, 'name': 'cots'}]

    for i, row in df.iterrows():

        images.append({
            "id": i,
            "file_name": f"video_{row['video_id']}_{row['video_frame']}.jpg",
            "height": 720,
            "width": 1280,
        })
        for bbox in row['annotations']:
            annotations.append({
                "id": annotion_id,
                "image_id": i,
                "category_id": 0,
                "bbox": list(bbox.values()),
                "area": bbox['width'] * bbox['height'],
                "segmentation": [],
                "iscrowd": 0
            })
            annotion_id += 1

    json_file = {'categories':categories, 'images':images, 'annotations':annotations}
    return json_file


def mmcfg_from_param(params):
    from mmcv import Config
    # model
    cfg = Config.fromfile(params['hyp_param']['base_file'])
    cfg.work_dir = str(params['output_dir'])
    cfg.seed = 2022
    cfg.gpu_ids = range(2)
    cfg.load_from = params['hyp_param']['load_from']    
    if params['hyp_param']['model_type'] == 'faster_rcnn':
        cfg.model.roi_head.bbox_head.num_classes = 1
        
        cfg.model.roi_head.bbox_head.loss_bbox.type = params['hyp_param']['loss_fnc']
        cfg.model.rpn_head.loss_bbox.type = params['hyp_param']['loss_fnc']
        if params['hyp_param']['loss_fnc'] == "GIoULoss":
            cfg.model.roi_head.bbox_head.reg_decoded_bbox = True
            cfg.model.rpn_head.reg_decoded_bbox = True
            
        
        cfg.model.train_cfg.rpn_proposal.nms.type = params['hyp_param']['nms']
        cfg.model.test_cfg.rpn.nms.type = params['hyp_param']['nms']
        cfg.model.test_cfg.rcnn.nms.type = params['hyp_param']['nms']
        
        cfg.model.train_cfg.rcnn.sampler.type = params['hyp_param']['sampler']            
        
    elif params['hyp_param']['model_type'] == 'swin':        
        pass # already changed
    elif params['hyp_param']['model_type'] == 'vfnet':
        cfg.model.bbox_head.num_classes = 1

    if params['hyp_param'].get("optimizer", cfg.optimizer.type) == "AdamW":
        cfg.optimizer = dict(
            type="AdamW",
            lr=params['hyp_param'].get("lr", cfg.optimizer.lr),
            weight_decay=params['hyp_param'].get(
                "weight_decay", cfg.optimizer.weight_decay
            ),
        )
    else:
        cfg.optimizer.lr = params['hyp_param'].get("lr", cfg.optimizer.lr)
        cfg.optimizer.weight_decay = params['hyp_param'].get(
                "weight_decay", cfg.optimizer.weight_decay)
    cfg.lr_config = dict(
            policy='CosineAnnealing', 
            by_epoch=False,
            warmup='linear', 
            warmup_iters= 1000, 
            warmup_ratio= 1/10,
            min_lr=1e-07)    
    
    # data
    cfg = add_data_pipeline(cfg, params)   
        
    cfg.runner.max_epochs = params['epochs']
    cfg.evaluation.start = 1
    cfg.evaluation.interval = 1
    cfg.evaluation.save_best='auto'
    cfg.evaluation.metric ='bbox'
    
    cfg.checkpoint_config.interval = -1
    cfg.log_config.interval = 500
    cfg.log_config.with_step = True
    cfg.log_config.by_epoch = True
    
    cfg.log_config.hooks =[dict(type='TextLoggerHook'),
                           dict(type='TensorboardLoggerHook')]    
    cfg.workflow = [('train',1)]
    
    logging.info(str(cfg))
    return cfg


def add_data_pipeline(cfg, params):
    cfg.dataset_type = 'COCODataset'
    cfg.classes = ('cots',)
    cfg.data_root = str(params['data_path'].resolve())
    params['aug_param']['img_scale'] = (params['img_size'], params['img_size'])
    cfg.img_scale = params['aug_param']['img_scale']
    cfg.dataset_type = 'CocoDataset'
    cfg.filter_empty_gt = False
    cfg.data.filter_empty_gt = False

    cfg.data.train.type = cfg.dataset_type
    cfg.data.train.classes = cfg.classes
    cfg.data.train.ann_file = str(params["cfg_dir"] / 'annotations_train.json')
    cfg.data.train.img_prefix = cfg.data_root + '/images/'
    cfg.data.train.filter_empty_gt = False

    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.classes = cfg.classes
    cfg.data.test.ann_file = str(params["cfg_dir"] / 'annotations_valid.json')
    cfg.data.test.img_prefix = cfg.data_root + '/images/'
    cfg.data.test.filter_empty_gt = False

    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.classes = cfg.classes
    cfg.data.val.ann_file = str(params["cfg_dir"] / 'annotations_valid.json')
    cfg.data.val.img_prefix = cfg.data_root + '/images/'    
    cfg.data.val.filter_empty_gt = False
    
    cfg.data.samples_per_gpu = params['batch'] // len(cfg.gpu_ids)
    cfg.data.workers_per_gpu = params['workers'] // len(cfg.gpu_ids)        
    
    # train pipeline  
    albu_train_transforms = get_albu_transforms(params['aug_param'], is_train=True)
    
    if params['aug_param']['use_mixup'] or params['aug_param']['use_mosaic']:
        train_pipeline = []
    else:
        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)]
    if params['aug_param']['use_mosaic']:
        train_pipeline.append(dict(type='Mosaic', img_scale=cfg.img_scale, pad_val=114.0))
    else:
        train_pipeline.append(dict(type='Resize', img_scale=cfg.img_scale, keep_ratio=False))
        
    train_pipeline = train_pipeline +[
        dict(type='Pad', size_divisor=32),
        dict(
            type='Albu',
            transforms=albu_train_transforms,
            bbox_params=dict(
                type='BboxParams',
                format='pascal_voc',
                label_fields=['gt_labels'],
                min_visibility=0.0,
                filter_lost_elements=True),
            keymap={
                'img': 'image',
                'gt_bboxes': 'bboxes'
            },
            update_pad_shape=False,
            skip_img_without_anno=False
        )]
    
    if params['aug_param']['use_mixup']:
        train_pipeline.append(dict(type='MixUp', img_scale=cfg.img_scale, ratio_range=(0.8, 1.6), pad_val=114.0))
        
    train_pipeline = train_pipeline +\
        [
        dict(type='Normalize', **cfg.img_norm_cfg),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', 
            keys=['img', 'gt_bboxes', 'gt_labels'],
            meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 
                        'scale_factor', 'img_norm_cfg')),
    ]
        
    val_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=cfg.img_scale,
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **cfg.img_norm_cfg),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
    ]
        
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=[cfg.img_scale],
            flip=[False],
            transforms=[
                dict(type='Resize', keep_ratio=False),
                dict(type='Pad', size_divisor=32),
                dict(type='RandomFlip', direction='horizontal'),
                dict(type='Normalize', **cfg.img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]    
    
    cfg.train_pipeline = train_pipeline
    cfg.val_pipeline = val_pipeline
    cfg.test_pipeline = test_pipeline
    
    
    if params['aug_param']['use_mixup'] or params['aug_param']['use_mosaic']:
        cfg.train_dataset = dict(
            type='MultiImageMixDataset',
            dataset=dict(
                type=cfg.dataset_type,
                classes=cfg.classes,
                ann_file=str(params["cfg_dir"] / 'annotations_train.json'),
                img_prefix=cfg.data_root + '/images/',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True)
                ],
                filter_empty_gt=False,
            ),
            pipeline=cfg.train_pipeline
        )
        cfg.data.train = cfg.train_dataset
    else:
        cfg.data.train.pipeline = cfg.train_pipeline
        
    cfg.data.val.pipeline = cfg.val_pipeline
    cfg.data.test.pipeline = cfg.test_pipeline 
    
    return cfg    


def find_ckp(output_dir):
    return glob(output_dir / "best*.pth")[0]