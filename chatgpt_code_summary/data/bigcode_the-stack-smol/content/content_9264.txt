"""Helper classes for easy recording of videos and images. The classes automatically
find a suitable name for the output file given a specific pattern.
"""
from abc import ABC, abstractmethod
import cv2
import os
from glob import glob

class Recorder(ABC):

    def __init__(self, path, file_pattern):
        existing = list(glob(os.path.join(path, file_pattern)))
        n = len(existing) + 1

        pat_left, pat_right = file_pattern.split('*')
        self.resource_path = os.path.join(path, f'{pat_left}{n}{pat_right}')


class VideoRecorder(Recorder):

    def __init__(self, path, frame_shape):
        super().__init__(path, 'vid_*.mp4')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.resource_path, fourcc, 60, (frame_shape[1], frame_shape[0]))

    def write(self, frame):
        self.writer.write(frame)

    def stop(self):
        self.writer.release()


class PictureRecorder(Recorder):

    def __init__(self, path, image):
        super().__init__(path, 'pic_*.jpg')
        print(self.resource_path)
        cv2.imwrite(self.resource_path, image)
