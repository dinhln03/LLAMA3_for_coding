import numpy as np
import cv2
from PIL import Image

img_form = "jpg"
img_out_dir = "./output_images"
vid_form = "mp4"
vid_out_dir = "./test_videos_output"

class array_image:
    def __init__(self):
        self.image = None
        self.binary_image = None

    def store(self, name):
        name = img_out_dir + "/" + name + "." + img_form
        print("Saving image: " + name)
        im = Image.fromarray(self.binary_image)
        im.save(name)

class color(array_image):
    def __init__(self, caller=None, color = "Gr"):
        threshold = {'R':(200, 255), 'G':(200, 255), 'B':(200, 255), 'H':(15, 100), 'L':(0,255), 'S':(90, 255), 'Gr':(200, 255)}
        self.available = False
        self.binary_available = False
        self.image = None
        self.binary_image = None
        self.caller = caller
        self.color = color
        self.threshold = threshold[self.color]

    def get(self, binary=False, masked=False, thresh=None):
        ret = 0
        if (self.available) & (thresh==None):
            if binary:
                if self.binary_available:
                    ret = self.binary_image
                else:
                    self.binary_image = self.color_select(color=self.color, binary=True)
                    self.binary_available = True
                    ret = self.binary_image
            else:
                ret = self.image
        else:
            self.image = self.color_select(color=self.color, binary=False)
            self.available = True
            if binary:
                self.binary_image = self.color_select(color=self.color, binary=True, thresh=None)
                self.binary_available = True
                ret = self.binary_image
            else:
                ret = self.image

        if masked:
            ret = self.caller.region_of_interest(ret)

        return ret

    def grayscale(self):
        """Applies the Grayscale transform
        This will return an image with only one color channel
        but NOTE: to see the returned image as grayscale
        (assuming your grayscaled image is called 'gray')
        you should call plt.imshow(gray, cmap='gray')"""
        return cv2.cvtColor(self.caller.image, cv2.COLOR_RGB2GRAY)
        # Or use BGR2GRAY if you read an image with cv2.imread()
        # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def color_select(self, color='R', binary = True, thresh=None):
        #image received is RGB  mpimg.imread
        img = np.copy(self.caller.image)
        RGB_colors = {'R':0, 'G':1, 'B':2}
        HLS_colors = {'H':0, 'L':1, 'S':2}
        if color in RGB_colors:
            channel = img[:,:,RGB_colors[color]]
        elif color in HLS_colors:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            channel = img[:, :, HLS_colors[color]]
        else:
            channel = self.grayscale()
        if binary:
            if not thresh:
                thresh = self.threshold

            binary_output = np.zeros_like(img[:,:,0])
            binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
            return binary_output
        else:
            return channel

