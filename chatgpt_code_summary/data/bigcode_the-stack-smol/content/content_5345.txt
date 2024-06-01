'''
Adapted from https://github.com/IntelligentQuadruped, with permission
Description: Module to connect to camera and retrieve RGB and depth data. Currently supports the Intel RealSense R200 Camera.
'''

import numpy as np
import logging
import time
import cv2
import matplotlib.pyplot as plt
from skimage.transform import rescale
from file_support import ensureDir
from os import path, makedirs

try:
    import pyrealsense as pyrs
except ImportError as error:
    logging.warning("cam.py: " + str(error))

class Camera:
    """
    Object to get data from R200
    """
    def __init__(self, max_depth = 4.0, save_images = False, \
        t_buffer = 5, output_dir = './Trials/'):
        """
        Intitalizes Camera object 
        """
        self.max_depth = max_depth
        self.save_images = save_images
        self.clock = time.time()
        self.t_buffer = t_buffer
        self.output_dir = output_dir
        self.data_dir = path.join(self.output_dir,"{}".format(time.strftime("%d_%b_%Y_%H:%M", time.localtime())))

        if self.save_images:	
            ensureDir(self.data_dir)
        pass

        np.warnings.filterwarnings('ignore')

    def connect(self):
        """
        Establishes connection to R200 camera
        """
        logging.info("Cam.py: connecting components")
        self.serv = pyrs.Service()
        self.dev = self.serv.Device(device_id=0, 
                                    streams=[\
                                        pyrs.stream.DepthStream(fps=60), pyrs.stream.ColorStream(fps=60)])

    def disconnect(self):
        """
        Disconnects from R200 camera
        """
        self.dev.stop()
        self.serv.stop()
        logging.info("Cam.py: camera disconnected")

    def getFrames(self, frames = 5, rgb = False):
        """
        Retrieves depth frames (and RGB if true) from R200 input, cleans and averages depth images
        """
        self.dev.wait_for_frames()

        # Convert depth to meters
        depth = self.dev.depth * self.dev.depth_scale
        col = self.dev.color

        if self.save_images and (time.time() - self.clock > self.t_buffer):
            np.save(path.join(self.data_dir,str(time.time())+"_d"),depth)
            np.save(path.join(self.data_dir,str(time.time())+"_c"),col)
            self.clock = time.time()

        for _ in range(frames-1):
            self.dev.wait_for_frames()
            # Convert depth to meters
            curr = self.dev.depth * self.dev.depth_scale
            depth = np.dstack((depth, curr))

        if frames != 1:
            depth = np.nanmean(depth, 2)

        depth[depth <= 0] = np.nan
        depth[depth > self.max_depth] = np.nan

        if rgb:
            return depth, col

        return depth

    def reduceFrame(self, depth, height_ratio = 0.5, sub_sample = 0.3, reduce_to = 'lower'):
        """
        Takes in a depth image and rescales it

        Args:
            height_ratio: Determines fraction of rows to keep
            sub_sample: Scaling factor for image
        """
        if (height_ratio > 1.0) or (height_ratio < 0.0)\
            or (sub_sample > 1.0) or (sub_sample < 0.0):
            print('height_ratio and sub_sample must be between 0 and 1')
            exit(1)
        
        depth_copy = depth.copy()
        height = depth_copy.shape[0]
        h = int(height_ratio*(height))
        cols_to_cut = 0

        # catches the case when all rows are kept
        if height_ratio == 1:
            d_short = depth_copy

        elif reduce_to == 'lower':
            d_short = depth_copy[(height - h):,\
                cols_to_cut:-(cols_to_cut+1)]

        elif reduce_to == 'middle_lower':
            upper_brdr = int(3*(height/4.0) - h/2)
            lower_brdr = upper_brdr + h
            d_short = depth_copy[upper_brdr:lower_brdr,\
                cols_to_cut:-(cols_to_cut+1)]

        elif reduce_to == 'middle':
            upper_brdr = int((height - h)/2.0)
            lower_brdr = upper_brdr + h
            d_short = depth_copy[upper_brdr:lower_brdr,\
                cols_to_cut:-(cols_to_cut+1)]

        elif reduce_to == 'middle_upper':
            upper_brdr = int((height/4.0) - h/2)
            lower_brdr = upper_brdr + h
            d_short = depth_copy[upper_brdr:lower_brdr,\
                cols_to_cut:-(cols_to_cut+1)]

        elif reduce_to == 'upper':
            d_short = depth_copy[:h, cols_to_cut:-(cols_to_cut+1)]

        d_short[d_short <= 0] = np.nan
        d_short[d_short > self.max_depth] = np.nan
        
        rescaled = rescale(d_short, sub_sample, mode='reflect', multichannel=False, anti_aliasing=True)

        return rescaled

def main():
    """
    Unit tests
    """
    max_depth = 4.0
    numFrames = 10
    # height_ratio of 0 crops 0 rows away
    height_ratio = 0.5
    sub_sample = 1
    # reduce_to argument can be: 'lower', 'middle_lower', 'middle', 'middle_upper', and 'upper'
    reduce_to = 'middle_lower'

    print('Program settings:')
    print('\tmax_depth: ' + str(max_depth))
    print('\tnumFrames: ' + str(numFrames))
    print('\theight_ratio: ' + str(height_ratio))
    print('\tsub_sample: ' + str(sub_sample))
    print('\treduce_to: ' + reduce_to)

    cam = Camera(max_depth = max_depth)
    cam.connect()
    time.sleep(2.5)

    t1 = time.time()
    d = cam.getFrames(numFrames)
    t2 = time.time()
    printStmt = 'Time to get {0} frames: ' + str(t2 - t1)
    print(printStmt.format(numFrames))
    d_small = cam.reduceFrame(d, height_ratio = height_ratio, sub_sample = sub_sample, reduce_to = reduce_to)

    # colormap:
    # https://matplotlib.org/tutorials/colors/colormaps.html

    # scaled depth
    plt.figure(figsize = (6, 7)) # figsize = width, height
    ax2 = plt.subplot(2, 1, 2)
    plt.imshow(d_small, cmap='gist_rainbow')
    plt.colorbar()
    plt.title('Scaled (height_ratio = {0}, sub_sample = {1})'.format(height_ratio, sub_sample))
    plt.grid()

    # original depth
    # plt.subplot(2, 1, 1, sharex=ax2, sharey=ax2)
    plt.subplot(2, 1, 1)
    plt.imshow(d, cmap='gist_rainbow')
    plt.colorbar()
    plt.title('Original')
    plt.grid()

    plt.subplots_adjust(hspace = 0.3)

    plt.show()
    cam.disconnect()

if __name__ == "__main__":
    main()