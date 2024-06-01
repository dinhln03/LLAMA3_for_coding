import os
import pandas
import numpy as np
from numpy.random import default_rng
import cv2
from time import time_ns
from datetime import datetime, timedelta
from PIL import Image


class Imagebot:

    def __init__(self, queue="./queue", sourcedir="./source", index="index.csv",
                 min_queue_length=240, images_to_autoadd=24):
        self.queue = queue
        self.sourcedir = sourcedir
        self.index = pandas.read_csv(index)
        self.min_queue_length = min_queue_length
        self.images_to_autoadd = images_to_autoadd

    def get_specific_image(self, key, frame):
        # 1. Find the first file in the source directory that matches the key
        #    If none, go with the first file in the source directory
        files = os.listdir(self.sourcedir)
        file = files[0]
        for file_check in files:
            if key in file_check:
                file = file_check
                break

        filepath = os.path.join(self.sourcedir, file)

        # 2. Extract the frame
        video = cv2.VideoCapture(filepath)
        video.set(1, frame)  # Set the frame
        ret, im = video.read()

        # 3. Return the result
        return im

    def get_random_image(self):
        # Returns the image data from a random clip in the source files
        # 1. Pick a clip (row) from the index
        clip = default_rng().integers(0, self.index.shape[0])

        # 2. Extract the data from the row
        key = self.index.iloc[clip]["key"]
        clip_start = self.index.iloc[clip]["clip_start"]
        clip_end = self.index.iloc[clip]["clip_end"]

        # 3. Pick a random frame from the clip
        frame = default_rng().integers(clip_start, clip_end+1)

        # 4. Return the result
        return self.get_specific_image(key, frame)

    @staticmethod
    def rgb_correction(im):
        # CV2 package switches the red and blue channels for some reason, correct them here
        b, g, r = Image.fromarray(im).split()
        image = Image.merge("RGB", (r, g, b))
        return image

    def populate_queue(self, n=1):
        # Add n images to the queue
        print("Retreiving", n, "images")
        start_time = time_ns()
        for i in np.arange(n)+1:
            im = self.get_random_image()
            image = self.rgb_correction(im)
            filename = "New File - " + str(time_ns()) + ".png"
            filepath = os.path.join(self.queue, filename)
            image.save(filepath)
            print(i, filepath)
        end_time = time_ns()
        delta = (end_time - start_time) / 1e9
        avg = delta / n
        print("Retreived", n, "images")
        print("    Total time:", np.round(delta, decimals=1), "seconds")
        print("  Average time:", np.round(avg, decimals=1), "seconds per image")

    def autopopulate_queue(self, min_queue_length=240, images_to_add=24):
        # Check the length of the queue, if it's below the specified threshold then run populate_queue()
        # Defaults are good for an hourly bot (Queue is at least 10 days long, add 24 images at a time)
        queue_length = len(os.listdir(self.queue))
        print("There are", queue_length, "images left in the queue.")
        if queue_length < min_queue_length:
            print("Queue length is below threshold (min", min_queue_length, "images)")
            self.populate_queue(images_to_add)

    def pop_from_queue(self, dryrun=False):
        # Return the raw image data from the first image in the queue & delete it
        # Use this method to post the data to Twitter or some other API
        files = os.listdir(self.queue)
        files.sort()
        file = files[0]
        filepath = os.path.join(self.queue, file)
        imagefile = open(filepath, "rb")
        imagedata = imagefile.read()
        print("Loaded data from", filepath)
        if not dryrun:
            os.remove(filepath)
            print("Removed", filepath)
        else:
            print("Dry run, did not remove", filepath)
        return imagedata

    def organize_queue(self, start=datetime.now(), interval=60, dryrun=False):
        # Rename the files in the queue with timestamps of when they will be posted
        # Default settings: Items in the queue are posted an hour apart, starting now

        # Get the queue and sort it
        files = os.listdir(self.queue)
        files.sort()

        # Loop through and rename the files
        for i in range(len(files)):
            stamp = start + timedelta(minutes=i*interval)
            stamp_str = stamp.strftime("%Y-%m-%d %H:%M")
            extension = os.path.splitext(files[i])[1]
            src = os.path.join(self.queue, files[i])
            dst = os.path.join(self.queue, stamp_str+extension)
            print("os.rename("+src+", "+dst+")")
            if not dryrun:
                os.rename(src, dst)
