#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import random

from typing import Sequence, Tuple, Dict, cast

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from tracking import TrackedDetection

ColorT = Tuple[int, int, int]
PointT = Tuple[int, int]


def labeled_rectangle(
        image: np.ndarray, start_point: PointT, end_point: PointT, label: str,
        rect_color: ColorT, label_color: ColorT, alpha: float = 0.85):
    (x1, y1), (x2, y2) = start_point, end_point

    roi = image[y1:y2, x1:x2]
    rect = np.ones_like(roi) * 255
    image[y1:y2, x1:x2] = cv.addWeighted(roi, alpha, rect, 1 - alpha, 0)

    font_face = cv.FONT_HERSHEY_COMPLEX_SMALL
    font_scale = 1
    font_thickness = 3

    (text_width, text_height), baseline = cv.getTextSize(
        label, font_face, font_scale, font_thickness)
    text_rect_end = (
        start_point[0] + text_width, start_point[1] + text_height + baseline)
    cv.rectangle(image, start_point, text_rect_end, rect_color, -1)
    
    # TODO Somehow calculate the shift.
    text_start_point = (start_point[0] + 1, start_point[1] + text_height + 3)
    cv.putText(
        image, label, text_start_point, font_face, font_scale, label_color,
        font_thickness, cv.LINE_AA)
    cv.putText(
        image, label, text_start_point, font_face, font_scale, (255, 255, 255),
        max(1, font_thickness - 2), cv.LINE_AA)
    cv.rectangle(image, start_point, end_point, rect_color, 2, cv.LINE_AA)


class TrackingVisualizer:
    def __init__(self, n_colors: int) -> None:
        assert n_colors > 0
        
        self.colors: Sequence[ColorT] = self.init_colors(n_colors, True)
        self.track_color: Dict[int, ColorT] = {}
    
    def draw_tracks(
            self, image: np.ndarray,
            tracks: Sequence[TrackedDetection]) -> None:
        for track in tracks:
            text = str(track.track_id)
            text_color = self._get_text_color()
            annotation_color = self._get_annotation_color(track)
            labeled_rectangle(
                image, track.box.top_left, track.box.bottom_right, text,
                annotation_color, text_color)
    
    def _get_text_color(self) -> ColorT:
        return (16, 16, 16)
    
    def _get_annotation_color(self, track: TrackedDetection) -> ColorT:
        color = self.track_color.get(track.track_id)
        if color is not None:
            return color
        color_pos = len(self.track_color) % len(self.colors)
        color = self.colors[color_pos]
        self.track_color[track.track_id] = color
        return cast(ColorT, color)
    
    @staticmethod
    def init_colors(n_colors: int, randomize: bool = False) -> Sequence[ColorT]:
        color_map = plt.cm.get_cmap('Spectral', n_colors)
        colors = [
            tuple(int(round(c * 255)) for c in color_map(i)[:3])
            for i in range(n_colors)]
        if randomize:
            random.shuffle(colors)
        return cast(Sequence[ColorT], colors)
