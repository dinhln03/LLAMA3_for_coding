#!/usr/bin/python

import sys
import os
import tkinter
import joblib
import pathlib

from PIL import Image, ImageTk
from PIL.ExifTags import TAGS
from pathlib import Path
from collections import deque

# Built based off of: https://github.com/Lexing/pyImageCropper

# ================================================================
#
# Module scope funcitons
#
# ================================================================


def get_current_folder():
    return str(pathlib.Path(__file__).parent.absolute())


def _get_filename(filepath):
    """ get filename from path """
    return str(Path(filepath).name)


def _scale_image(img, maxLen, maxHeight):
    """ scale image to under the specified maxLen and maxHeight """
    scale = 1
    resized_img = img.copy()

    # if > maxLen width, resize to maxLen
    if resized_img.size[0] > maxLen:
        resize = resized_img.size[0] / maxLen
        answer = (int(resized_img.size[0] / resize), int(resized_img.size[1] / resize))
        scale = resize
        resized_img = resized_img.resize(answer, Image.ANTIALIAS)

    # if > maxHeight height, resize to maxHeight
    if resized_img.size[1] > maxHeight:
        resize = (resized_img.size[1] / maxHeight)
        answer = (int(resized_img.size[0] / resize), int(resized_img.size[1] / resize))
        scale = scale * resize
        resized_img = resized_img.resize(answer, Image.ANTIALIAS)

    return resized_img, scale


def _point_on_image(point, img):
    """ check if point is on the image """
    x, y = point
    if x >= 0 and x <= img.size[0]:
        if y >= 0 and y <= img.size[1]:
            return True
    return False

# ================================================================
#
# Module scope classes
#
# ================================================================


class _DataStore:
    """
    Stores data about the current state
    """
    FOLDER_PROG_KEY = "FOLDER_PROG"

    KEY_TO_STORE = {FOLDER_PROG_KEY: {}}

    def __init__(self, filepath):
        self._filepath = filepath
        self._store = self._load_store()
        for key in _DataStore.KEY_TO_STORE:
            self.build_store(key)

    def save_value(self, key, value):
        self._store[key] = value

    def get_value(self, key):
        if key in self._store:
            return self._store[key]

    def build_store(self, key):
        if key in self._store:
            return self._store[key]
        else:
            self._store[key] = _DataStore.KEY_TO_STORE[key]
            return self._store[key]

    def save_store(self, delete=False):
        self._delete_store() if delete else self._write_store()

    def _load_store(self):
        if os.path.exists(self._filepath):
            return joblib.load(self._filepath)
        else:
            return {}

    def _delete_store(self):
        os.remove(self._filepath)

    def _write_store(self):
        joblib.dump(self._store, self._filepath, compress=9)


class ImageCanvas:
    """
    Image canvas area of the GUI
    """

    def __init__(self, tkRoot, height=800, width=1200, boxBasePx=32):

        # vals
        self.canvas_image = None    # TK image on the canvas
        self.canvasHeight = height  # TK canvas height
        self.canvasWidth = width    # TK canvas width
        self.rectangle = None       # TK rectangle on the canvas
        self.box = [0, 0, 0, 0]     # Need to turn this into array of points
        self.boxBasePx = boxBasePx  # base of the rectangle to build crop box
        self.img = None             # curr origional image
        self.resized_img = None     # curr resized image
        self.movingCrop = False     # crop is currently moving
        self.lastLocation = [0, 0]  # last location the mouse was clicked down at
        self.scale = 1              # scale image was reduced by
        self.currImage = None       # current TK image in canvas. Need reference to it, or garbage coll cleans it up.

        # objs
        self.canvas = tkinter.Canvas(tkRoot,
                                     highlightthickness=0,
                                     bd=0)

        # movement button binds
        tkRoot.bind("<Button-1>", self._on_mouse_down)
        tkRoot.bind("<ButtonRelease-1>", self._on_mouse_release)
        tkRoot.bind("<B1-Motion>", self._on_mouse_move)

    # primary methods
    # ================================================================

    def roll_image(self, imgLoc):
        """ changes canvas to a new image """

        # open image
        self.img = Image.open(imgLoc)

        # scale to fit area
        self.resized_img, self.scale = _scale_image(self.img, self.canvasWidth, self.canvasHeight)
        self.currImage = ImageTk.PhotoImage(self.resized_img)

        # setup canvas
        self.canvas.delete("all")
        self.canvas.config(width=self.resized_img.size[0], height=self.resized_img.size[1])
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tkinter.NW, image=self.currImage)
        self.canvas.pack(fill=tkinter.BOTH, expand=tkinter.YES)

        # build pre-req objs
        self._build_crop_box()
        self._refresh_crop_rectangle()

    def crop_image(self):

        # scale box back from the viewed area
        box = [self.box[0] * self.scale, self.box[1] * self.scale,
               self.box[2] * self.scale, self.box[3] * self.scale]

        # make the crop
        cropped = self.img.crop(box)

        # error?
        if cropped.size[0] == 0 and cropped.size[1] == 0:
            print('image has no size!!!')

        # edge case from resizing. Should change this to resize crop to fix diff? Only occcurs every 1 in like 600 so far
        # Possible fix: if remainder above half of base, then resize HIGHER else resize LOWER?
        if not((cropped.size[0] * cropped.size[1]) % self.boxBasePx == 0):
            return None

        return cropped

    # helper methods
    # ================================================================

    def _build_crop_box(self):
        """ creates the box for the crop rectangle x1,y1,x2,y2"""
        # get min side length of image
        boxMax = min(self.resized_img.size[0], self.resized_img.size[1])
        # (length of side - (remainder of side left after removing area divisible by boxBase))
        newImgLen = (boxMax - (boxMax % (self.boxBasePx / self.scale)))
        # build box from side length
        self.box = [0, 0, newImgLen, newImgLen]

    def _refresh_crop_rectangle(self, deltaX=0, deltaY=0):
        """ re-builds the crop rectangle based on the specified box """
        if self.rectangle and deltaX > 0 or deltaY > 0:
            self.canvas.move(self.rectangle, deltaX, deltaY)
        else:
            self.canvas.delete(self.rectangle)
            self.rectangle = self.canvas.create_rectangle(self.box[0],
                                                          self.box[1],
                                                          self.box[2],
                                                          self.box[3],
                                                          outline='red',
                                                          width=2)

    # movement methods
    # ================================================================

    def _on_mouse_down(self, event):
        """ if mouse clicked on crop area, allow moving crop """
        if event.x >= self.box[0] and event.x <= self.box[2]:
            if event.y >= self.box[1] and event.y <= self.box[3]:
                self.movingCrop = True
                self.lastLocation = [event.x, event.y]

    def _on_mouse_release(self, event):
        """ stop allowing movement of crop area """
        self._on_mouse_move(event)
        self.movingCrop = False

    def _on_mouse_move(self, event):
        """ move crop along with the user's mouse """
        if self.movingCrop:
            if _point_on_image((event.x, event.y), self.resized_img):

                # build delta from last spot
                deltaX = event.x - self.lastLocation[0]
                deltaY = event.y - self.lastLocation[1]

                # force area of box to conform to area of image
                if self.box[0] + deltaX < 0:
                    deltaX = 0

                if self.box[1] + deltaY < 0:
                    deltaY = 0

                if self.box[2] + deltaX > self.resized_img.size[0]:
                    deltaX = self.box[2] - self.resized_img.size[0]

                if self.box[3] + deltaY > self.resized_img.size[1]:
                    deltaY = self.box[3] - self.resized_img.size[1]

                # calc
                self.box = [self.box[0] + deltaX, self.box[1] + deltaY,
                            self.box[2] + deltaX, self.box[3] + deltaY]

                # move box
                self._refresh_crop_rectangle(deltaX, deltaY)
                self.lastLocation = [event.x, event.y]


class ImageCropper:
    """
    Main module class
    """

    def __init__(self, inputDir, outputDir, canvasHeight=800, canvasWidth=1200, cropBasePx=32):

        # vals
        self.inputDir = inputDir
        self.outputDir = outputDir
        self.currImage = None
        self.fileQueue = deque()
        self.queueIndex = 0
        self.doneSet = set()
        self.canvasHeight = canvasHeight
        self.canvasWidth = canvasWidth
        self.cropBasePx = cropBasePx

        # objs
        self._datasource = _DataStore(get_current_folder() + '/dataStore')
        self._folderProgDict = self._datasource.get_value(_DataStore.FOLDER_PROG_KEY)
        self.tkroot = tkinter.Tk()
        self.imageCanvas = ImageCanvas(self.tkroot)

        # movement button binds
        self.tkroot.bind("<Key>", self._on_key_down)

    # primary methods
    # ================================================================

    def run(self):
        self._setup_store()
        self._pull_files()
        self._roll_image()
        self.tkroot.geometry(str(self.canvasWidth) + 'x' + str(self.canvasHeight))
        self.tkroot.mainloop()

    def _pull_files(self):
        if not os.path.isdir(self.inputDir):
            raise IOError(self.inputDir + ' is not a directory')
        files = os.listdir(self.inputDir)
        if len(files) == 0:
            print('No files found in ' + self.inputDir)
        for filename in files:
            self.fileQueue.append(os.path.join(self.inputDir, filename))

    def _roll_image(self):
        breakOut = False
        while True and not(breakOut):
            if self.fileQueue:
                self.currImage = self.fileQueue.popleft()
                if not(self.currImage in self.doneSet):
                    print('Index in queue ' + str(self.queueIndex))
                    try:
                        self.imageCanvas.roll_image(self.currImage)
                        breakOut = True
                    except IOError:
                        print('Ignore: ' + self.currImage + ' cannot be opened as an image')
                        breakOut = False
                self.queueIndex += 1
            else:
                breakOut = True
                self.tkroot.quit()
        self.tkroot.update()

    # helper methods
    # ================================================================

    def _setup_store(self):
        if self.inputDir in self._folderProgDict:
            self.doneSet = self._folderProgDict[self.inputDir]
        else:
            self._folderProgDict[self.inputDir] = self.doneSet

    def _save_image(self, img):
        if img:
            outputName = _get_filename(self.currImage)
            outputLoc = self.outputDir + '/' + outputName[:outputName.rfind('.')] + '_cropped'
            img.save(outputLoc + '.png', 'png')

    # movement methods
    # ================================================================

    def _on_key_down(self, event):
        if event.char == ' ':
            self._save_image(self.imageCanvas.crop_image())
            self.doneSet.add(self.currImage)
            self._datasource.save_store()
            self._roll_image()
        elif event.char == 's':
            self.doneSet.add(self.currImage)
            self._datasource.save_store()
            self._roll_image()
        elif event.char == 'q':
            self.tkroot.destroy()