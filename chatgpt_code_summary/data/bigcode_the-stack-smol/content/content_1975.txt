"""

This is the runner of the entire eva.jvc system.

Version 1,
the steps for the entire pipeline are as follows:
1. preprocessor -- get rep indices, save children metadata
2. encoder -- encode video by forcing i-frames (also modify the i-frame skip rate)
3. decoder -- using metadata, select the i-frames you want to decode.
              If the user wants more frames than the number of i frames, then I guess we have to decode the entire video??

@Jaeho Bang

"""

import os

from eva_storage.jvc.preprocessor import *
from eva_storage.jvc.encoder import *
from eva_storage.jvc.decoder import *

from loaders.seattle_loader import SeattleLoader
from timer import Timer


class JVCRunner_v2:

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.compressor = Compressor()
        self.decompressor = Decompressor()
        self.video_loader = SeattleLoader()

    def encode(self, path_to_video):
        video_filename = os.path.basename(path_to_video)
        ###TODO: eliminate the extension
        video_filename = video_filename.split('.')[0]
        images, metadata = self.video_loader.load_images(
            path_to_video)  ## we might need metadata such as fps, frame_width, frame_height, fourcc from here
        rep_indices = self.preprocessor.run(images, video_filename)
        self.compressor.run(images, rep_indices, metadata)

    def decode(self, path_to_video, number_of_samples=None):
        images = self.decompressor.run(path_to_video, number_of_samples)

        return images


if __name__ == "__main__":
    timer = Timer()  ##TODO: use the timer to run the pipeline
    preprocessor = Preprocessor()
    compressor = Compressor()
    decompressor = Decompressor()

    video_loader = SeattleLoader()
    images = video_loader.load_images()
    meta_data = preprocessor.run(images)
    save_directory = compressor.run(images, meta_data)

    number_of_frames = 100  ## we can change this to whatever number we want
    images_jvc = decompressor.run(save_directory, number_of_frames)


