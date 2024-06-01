from typing import List, Tuple, Union

import numpy as np
import torch

import pytorch_lightning as pl


def calc_area(bbox: np.ndarray):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def calc_bbox_overlap_union_iou(pred: np.ndarray or None, teacher: np.ndarray) -> Tuple[float, float, float]:
    """
    :param pred: ndarray (4, )
    :param teacher: ndarray (4, )
    :return: overlap, union, iou
    """
    teacher_area = (teacher[2] - teacher[0]) * (teacher[3] - teacher[1])
    if pred is None:
        return 0.0, teacher_area, 0.0

    pred_area = (pred[2] - pred[0]) * (pred[3] - pred[1])

    intersection_width = np.maximum(np.minimum(pred[2], teacher[2]) - np.maximum(pred[0], teacher[0]), 0)
    intersection_height = np.maximum(np.minimum(pred[3], teacher[3]) - np.maximum(pred[1], teacher[1]), 0)

    overlap = intersection_width * intersection_height
    union = teacher_area + pred_area - overlap
    iou = overlap / union
    return overlap, union, iou


class DetectionIoU(pl.metrics.Metric):
    def __init__(self, n_classes: int, by_classes: bool = False):
        super().__init__(compute_on_step=False)
        self._n_classes = n_classes
        self._by_classes = by_classes
        self.add_state("image_count_by_classes", default=torch.tensor([0. for _ in range(n_classes)]),
                       dist_reduce_fx="sum")
        self.add_state("total_iou_by_classes", default=torch.tensor([0. for _ in range(n_classes)]),
                       dist_reduce_fx="sum")

    def update(self, preds: List[np.ndarray], targets: Union[np.ndarray, torch.Tensor]) -> None:
        """
        :param preds: Sorted by score. (Batch size, bounding boxes by batch, 5(x_min, y_min, x_max, y_max, label))
        :param targets: (batch size, bounding box count, 5(x_min, y_min, x_max, y_max, label))
        :return:
        """
        targets = targets.cpu().detach().numpy() if isinstance(targets, torch.Tensor) else targets
        # 全探索だと遅いのでクラスごとにまとめておく
        preds_by_class = []
        for pred_bboxes in preds:
            pred_by_class = [[] for _ in range(self._n_classes)]
            for pred_bbox in pred_bboxes:
                pred_by_class[int(pred_bbox[4])].append(pred_bbox)
            preds_by_class.append(pred_by_class)

        for i in range(targets.shape[0]):  # Explore every batch.
            bbox_annotations = targets[i, :, :]
            # Exclude invalid label annotation.
            bbox_annotations = bbox_annotations[bbox_annotations[:, 4] >= 0]

            pred_by_class = preds_by_class[i]

            """
            1画像でラベルごとに計算.
            ラベルごとの面積合計/overlapを計算
            1画像ごとにIoU算出、最終的に画像平均を算出
            """

            total_area_by_classes = [0 for _ in range(self._n_classes)]
            total_overlap_by_classes = [0 for _ in range(self._n_classes)]
            is_label_appeared = [False for _ in range(self._n_classes)]
            for bbox_annotation in bbox_annotations:

                label = int(bbox_annotation[4])
                total_area_by_classes[label] += calc_area(bbox_annotation)
                pred_bboxes = pred_by_class[label]

                if pred_bboxes is None or len(pred_bboxes) == 0:
                    continue

                # Calculate area and overlap by class.
                for pred_bbox in pred_bboxes:
                    overlap, _, _ = calc_bbox_overlap_union_iou(pred_bbox, bbox_annotation)
                    total_overlap_by_classes[label] += overlap
                    if is_label_appeared[label]:
                        continue
                    total_area_by_classes[label] += calc_area(pred_bbox)
                is_label_appeared[label] = True

            for label in range(self._n_classes):
                # Not exist label in this data.
                if total_area_by_classes[label] <= 0:
                    continue
                self.total_iou_by_classes[label] += total_overlap_by_classes[label] / (
                        total_area_by_classes[label] - total_overlap_by_classes[label])
                self.image_count_by_classes[label] += 1

    def compute(self):
        epsilon = 1e-8
        iou_by_classes = self.total_iou_by_classes / (self.image_count_by_classes + epsilon)
        if self._by_classes:
            return iou_by_classes
        return torch.mean(iou_by_classes)


class RecallPrecision(pl.metrics.Metric):
    def __init__(self, n_classes: int, by_classes: bool = False):
        super().__init__(compute_on_step=False)
        self._n_classes = n_classes
        self._by_classes = by_classes
        self.add_state("tp_by_classes", default=torch.tensor([0 for _ in range(n_classes)]), dist_reduce_fx="sum")
        self.add_state("fp_by_classes", default=torch.tensor([0 for _ in range(n_classes)]), dist_reduce_fx="sum")
        self.add_state("fn_by_classes", default=torch.tensor([0 for _ in range(n_classes)]), dist_reduce_fx="sum")

    def update(self, preds: List[np.ndarray], targets: Union[np.ndarray, torch.Tensor]) -> None:
        """
        :param preds: Sorted by score. (Batch size, bounding boxes by batch, 5(x_min, y_min, x_max, y_max, label))
        :param targets: (batch size, bounding box count, 5(x_min, y_min, x_max, y_max, label))
        :return:
        """
        targets = targets.cpu().detach().numpy() if isinstance(targets, torch.Tensor) else targets
        # 全探索だと遅いのでクラスごとにまとめておく
        preds_by_class = []
        for pred_bboxes in preds:
            pred_by_class = [[] for _ in range(self._n_classes)]
            for pred_bbox in pred_bboxes:
                pred_by_class[int(pred_bbox[4])].append(pred_bbox)
            preds_by_class.append(pred_by_class)

        for i in range(targets.shape[0]):
            bbox_annotations = targets[i, :, :]
            # Exclude invalid label annotation.
            bbox_annotations = bbox_annotations[bbox_annotations[:, 4] >= 0]

            pred_by_class = preds_by_class[i]

            applied_bbox_count_by_classes = [0 for _ in range(self._n_classes)]
            for bbox_annotation in bbox_annotations:
                label = int(bbox_annotation[4])
                pred_bboxes = pred_by_class[label]

                if pred_bboxes is None or len(pred_bboxes) == 0:
                    self.fn_by_classes[label] += 1
                    continue
                # Explore max iou of bbox_annotation
                is_matched = False
                for pred_bbox in pred_bboxes:
                    overlap, union, iou = calc_bbox_overlap_union_iou(pred_bbox, bbox_annotation)
                    if iou >= 0.5:
                        applied_bbox_count_by_classes[label] += 1
                        self.tp_by_classes[label] += 1
                        is_matched = True
                        break
                if not is_matched:
                    self.fn_by_classes[label] += 1

            for label in range(self._n_classes):
                self.fp_by_classes[label] += len(pred_by_class[label]) - applied_bbox_count_by_classes[label]

    def compute(self):
        epsilon = 1e-8
        recall = self.tp_by_classes / (self.tp_by_classes + self.fn_by_classes + epsilon)
        precision = self.tp_by_classes / (self.tp_by_classes + self.fp_by_classes + epsilon)
        f_score = 2. * recall * precision / (recall + precision + epsilon)
        if self._by_classes:
            return recall, precision, f_score
        return torch.mean(recall), torch.mean(precision), torch.mean(f_score)


class MeanAveragePrecision(pl.metrics.Metric):
    def __init__(self, n_classes: int, by_classes=False):
        super().__init__(compute_on_step=False)
        self._n_classes = n_classes
        # TODO want to implement using add_state
        self.fp_list_by_classes = [[] for _ in range(n_classes)]
        self.tp_list_by_classes = [[] for _ in range(n_classes)]
        self.score_list_by_classes = [[] for _ in range(n_classes)]
        self.num_annotations_by_classes = [0 for _ in range(n_classes)]
        # self.add_state("fp_list_by_classes", default=[[] for _ in range(n_classes)], dist_reduce_fx="cat")
        # self.add_state("tp_list_by_classes", default=[[] for _ in range(n_classes)], dist_reduce_fx="cat")
        # self.add_state("score_list_by_classes", default=[[] for _ in range(n_classes)], dist_reduce_fx="cat")
        # self.add_state("num_annotations_by_classes", default=[0 for _ in range(n_classes)], dist_reduce_fx="cat")
        self._by_classes = by_classes

    def update(self, preds: List[np.ndarray], targets: Union[np.ndarray, torch.Tensor]) -> None:
        """
        :param preds: Sorted by score. (Batch size, bounding boxes by batch, 5(x_min, y_min, x_max, y_max, label))
        :param targets: (batch size, bounding box count, 5(x_min, y_min, x_max, y_max, label))
        :return:
        """
        targets = targets.cpu().detach().numpy() if isinstance(targets, torch.Tensor) else targets
        for i in range(len(preds)):
            pred_bboxes, target_bboxes = preds[i], targets[i]
            # exclude invalid annotations.
            target_bboxes = target_bboxes[target_bboxes[:, 4] >= 0]
            self._update_num_annotations(target_bboxes)
            self._update_tp_fp_score(pred_bboxes, target_bboxes)

    def compute(self):
        ap_by_classes = [0 for _ in range(self._n_classes)]
        for label in range(self._n_classes):
            num_annotations = self.num_annotations_by_classes[label]
            tp_list, fp_list = np.array(self.tp_list_by_classes[label]), np.array(self.fp_list_by_classes[label])
            scores = np.array(self.score_list_by_classes[label])
            indices = np.argsort(-scores)
            # sort by score
            tp_list, fp_list = tp_list[indices], fp_list[indices]
            # cumulative sum
            tp_list, fp_list = np.cumsum(tp_list), np.cumsum(fp_list)

            if num_annotations == 0:
                ap_by_classes[label] = 0
                continue
            recall_curve = tp_list / num_annotations
            precision_curve = tp_list / np.maximum(tp_list + fp_list, np.finfo(np.float64).eps)
            ap_by_classes[label] = self._compute_average_precision(recall_curve, precision_curve)
        return ap_by_classes if self._by_classes else sum(ap_by_classes) / len(ap_by_classes)

    def _update_tp_fp_score(self, pred_bboxes: np.ndarray, target_bboxes: np.ndarray):
        """
        :param pred_bboxes: (N, 6(xmin, ymin, xmax, ymax, class, score))
        :param target_bboxes: (N, 5(xmin, ymin, xmax, ymax, class))
        """
        detected_indices = []
        for i in range(pred_bboxes.shape[0]):
            pred_label, pred_score = int(pred_bboxes[i][4]), pred_bboxes[i][5]
            matched = False
            for j in filter(lambda k: int(target_bboxes[k][4]) == pred_label and k not in detected_indices,
                            range(target_bboxes.shape[0])):
                overlap, union, iou = calc_bbox_overlap_union_iou(pred_bboxes[i], target_bboxes[j])
                if iou >= 0.5:
                    detected_indices.append(j)
                    self.fp_list_by_classes[pred_label].append(0)
                    self.tp_list_by_classes[pred_label].append(1)
                    matched = True
                    break
            if not matched:
                self.fp_list_by_classes[pred_label].append(1)
                self.tp_list_by_classes[pred_label].append(0)
            self.score_list_by_classes[pred_label].append(pred_score)

    def _update_num_annotations(self, target_bboxes: np.ndarray):
        """
        :param target_bboxes: (N, 5(xmin, ymin, xmax, ymax, class))
        """
        counts = list(map(lambda i: np.count_nonzero(target_bboxes[:, 4] == i), range(self._n_classes)))
        self.num_annotations_by_classes = list(
            map(lambda i: counts[i] + self.num_annotations_by_classes[i], range(self._n_classes)))

    def _compute_average_precision(self, recall_curve: np.ndarray, precision_curve: np.ndarray):
        # Reference by https://github.com/toandaominh1997/EfficientDet.Pytorch/blob/master/eval.py
        assert recall_curve.ndim == 1 and precision_curve.ndim == 1
        # correct AP calculation
        # first append sentinel values at the end
        mean_recall = np.concatenate(([0.], recall_curve, [1.]))
        mean_precision = np.concatenate(([0.], precision_curve, [0.]))

        # compute the precision envelope
        for i in range(mean_precision.size - 1, 0, -1):
            mean_precision[i - 1] = np.maximum(mean_precision[i - 1], mean_precision[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mean_recall[1:] != mean_recall[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mean_recall[i + 1] - mean_recall[i]) * mean_precision[i + 1])
        return ap

    def reset(self):
        self.fp_list_by_classes = [[] for _ in range(self._n_classes)]
        self.tp_list_by_classes = [[] for _ in range(self._n_classes)]
        self.score_list_by_classes = [[] for _ in range(self._n_classes)]
        self.num_annotations_by_classes = [0 for _ in range(self._n_classes)]
