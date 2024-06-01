"""
MIT License

Copyright (c) 2019 YangYun
Copyright (c) 2020 Việt Hùng
Copyright (c) 2020-2021 Hyeonki Hong <hhk7734@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from typing import List

import cv2
import numpy as np
from tensorflow.keras.utils import Sequence

from .augmentation import mosaic
from ...common import (
    media,
    convert_dataset_to_ground_truth as _convert_dataset_to_ground_truth,
)
from ...common.config import YOLOConfig
from ...common.parser import parse_dataset

_AUGMETATION_CACHE_SIZE = 50


class YOLODataset(Sequence):
    def __init__(
        self,
        config: YOLOConfig,
        dataset_list: str,
        dataset_type: str = "converted_coco",
        image_path_prefix: str = "",
        training: bool = True,
    ):
        self.dataset = parse_dataset(
            dataset_list=dataset_list,
            dataset_type=dataset_type,
            image_path_prefix=image_path_prefix,
        )
        self._metayolos = []
        if config.layer_count["yolo"] > 0:
            for i in range(config.layer_count["yolo"]):
                self._metayolos.append(config.find_metalayer("yolo", i))
        elif config.layer_count["yolo_tpu"] > 0:
            for i in range(config.layer_count["yolo_tpu"]):
                self._metayolos.append(config.find_metalayer("yolo_tpu", i))
        else:
            raise RuntimeError(
                "YOLODataset: model does not have a yolo or yolo_tpu layer"
            )

        self._metanet = config.net
        self._metayolos_np = np.zeros(
            (len(self._metayolos), 7 + len(self._metayolos[-1].mask)),
            dtype=np.float32,
        )
        for i, metayolo in enumerate(self._metayolos):
            self._metayolos_np[i, 0] = metayolo.height
            self._metayolos_np[i, 1] = metayolo.width
            self._metayolos_np[i, 2] = metayolo.channels
            self._metayolos_np[i, 3] = metayolo.classes
            self._metayolos_np[i, 4] = metayolo.label_smooth_eps
            self._metayolos_np[i, 5] = metayolo.max
            self._metayolos_np[i, 6] = metayolo.iou_thresh
            for j, mask in enumerate(metayolo.mask):
                self._metayolos_np[i, 7 + j] = mask

        self._anchors_np = np.zeros(
            len(self._metayolos[-1].anchors) * 2, dtype=np.float32
        )
        for i, anchor in enumerate(self._metayolos[-1].anchors):
            self._anchors_np[2 * i] = anchor[0] / self._metanet.width
            self._anchors_np[2 * i + 1] = anchor[1] / self._metanet.height

        # Data augmentation ####################################################

        self._augmentation: List[str] = []
        if config.net.mosaic:
            self._augmentation.append("mosaic")

        if training and len(self._augmentation) > 0:
            self._augmentation_batch = int(config.net.batch * 0.3)
            self._training = True
        else:
            self._augmentation_batch = 0
            self._training = False

        self._augmentation_cache = [
            self._get_dataset(i) for i in range(_AUGMETATION_CACHE_SIZE)
        ]
        self._augmentation_cache_index = 0

    def _convert_dataset_to_ground_truth(self, dataset_bboxes):
        """
        @param `dataset_bboxes`: [[b_x, b_y, b_w, b_h, class_id], ...]

        @return `groud_truth_one`:
            [Dim(yolo.h, yolo.w, yolo.c + len(mask))] * len(yolo)
        """
        return _convert_dataset_to_ground_truth(
            dataset_bboxes, self._metayolos_np, self._anchors_np
        )

    def _convert_dataset_to_image_and_bboxes(self, dataset):
        """
        @param dataset: [image_path, [[x, y, w, h, class_id], ...]]

        @return image, bboxes
            image: 0.0 ~ 1.0, Dim(1, height, width, channels)
        """
        # pylint: disable=bare-except
        try:
            image = cv2.imread(dataset[0])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            return None, None

        resized_image, resized_bboxes = media.resize_image(
            image,
            target_shape=self._metanet.input_shape,
            ground_truth=dataset[1],
        )
        resized_image = np.expand_dims(resized_image / 255.0, axis=0)

        return resized_image, resized_bboxes

    def _get_dataset(self, index: int):
        offset = 0
        for offset in range(5):
            image, bboxes = self._convert_dataset_to_image_and_bboxes(
                self.dataset[(index + offset) % len(self.dataset)]
            )
            if image is None:
                offset += 1
            else:
                return image, bboxes

        raise FileNotFoundError("Failed to find images")

    def __getitem__(self, index):
        """
        @return
            `images`: Dim(batch, height, width, channels)
            `groud_truth_one`:
                [Dim(batch, yolo.h, yolo.w, yolo.c + len(mask))] * len(yolo)
        """

        batch_x = []
        # [[gt_one, gt_one, ...],
        #  [gt_one, gt_one, ...], ...]
        batch_y = [[] for _ in range(len(self._metayolos))]

        start_index = index * self._metanet.batch

        for i in range(self._metanet.batch - self._augmentation_batch):
            image, bboxes = self._get_dataset(start_index + i)
            self._augmentation_cache[self._augmentation_cache_index] = (
                image,
                bboxes,
            )
            self._augmentation_cache_index = (
                self._augmentation_cache_index + 1
            ) % _AUGMETATION_CACHE_SIZE

            batch_x.append(image)
            ground_truth = self._convert_dataset_to_ground_truth(bboxes)
            for j in range(len(self._metayolos)):
                batch_y[j].append(ground_truth[j])

        for i in range(self._augmentation_batch):
            augmentation = self._augmentation[
                np.random.randint(0, len(self._augmentation))
            ]

            image = None
            bboxes = None
            if augmentation == "mosaic":
                image, bboxes = mosaic(
                    *[
                        self._augmentation_cache[
                            np.random.randint(
                                0,
                                _AUGMETATION_CACHE_SIZE,
                            )
                        ]
                        for _ in range(4)
                    ]
                )

            batch_x.append(image)
            ground_truth = self._convert_dataset_to_ground_truth(bboxes)
            for j in range(len(self._metayolos)):
                batch_y[j].append(ground_truth[j])

        return np.concatenate(batch_x, axis=0), [
            np.stack(y, axis=0) for y in batch_y
        ]

    def __len__(self):
        return len(self.dataset) // (
            self._metanet.batch - self._augmentation_batch
        )
