from tkinter import *
# import math

# https://www.youtube.com/watch?v=r5EQCSW_rLQ  pyramid math formulas TIME=3:55

class Pyramid:

    # contants
    BLOCK_HEIGHT = 1.5 # meters
    BLOCK_WIDTH = 2 # meters
    BLOCK_LENGTH = 2.5 # meters
    BLOCK_WEIGHT = 15000 # kg

    # __init__ is Python's constructor method
    def __init__(self, pyramidSideLength, pyramidHeight):
        self.pyramidSideLength = pyramidSideLength
        self.pyramidHeight = pyramidHeight

    # processing
    def calculateBlockVolume(self, height, width, length):
        return height * width * length

    def calculateGroundArea(self, length):
        return length ** 2

    def calculatePyramidVolume(self, groundArea, height):
        return round((groundArea / 3) * height)

    def countBlocks(self, pyramidVolume, blockVolume):
        return round(pyramidVolume / blockVolume)

    def calculateMass(self, blocks, weight):
        return blocks * weight

    # this type of function might not be suitable for inside a class, but going with it for now
    def createNewPyramid(self):
        # create superscript for displaying exponents
        superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        displayMetersSquared = 'm2'.translate(superscript)
        displayMetersCubed = 'm3'.translate(superscript)

        # storing function output in variables for program readability
        blockVolume = self.calculateBlockVolume(Pyramid.BLOCK_HEIGHT, Pyramid.BLOCK_WIDTH, Pyramid.BLOCK_LENGTH)
        groundAreaCovered = self.calculateGroundArea(self.pyramidSideLength)
        pyramidVolume = self.calculatePyramidVolume(groundAreaCovered, self.pyramidHeight)
        countOfBlocks = self.countBlocks(pyramidVolume, blockVolume)
        mass = self.calculateMass(countOfBlocks, Pyramid.BLOCK_WEIGHT)

        # build nicely formatted answer for display
        displayAnswer = '\n' + \
            'Ground Area Covered = {:,} {}'.format(groundAreaCovered, displayMetersSquared) + '\n' + \
            'Pyramid Volume = {:,.0f} {}'.format(pyramidVolume, displayMetersCubed) + '\n' + \
            'Blocks = {:,}'.format(countOfBlocks) + '\n' + \
            'Mass = {:,} kg'.format(mass) + '\n\n' + \
            '*Pyramid is not to scale.'

        return displayAnswer

class GridLines:
    
    def __init__(self, canvas, canvas_width, canvas_height, grid_space):
        self.canvas = canvas
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.grid_space = grid_space

    def __vertical_lines__(self):
        for i in range(0, self.canvas_height, self.grid_space):
            self.canvas.create_line(i, 0, i, self.canvas_height, fill='thistle1')

    def __horizontal_lines__(self):
        for i in range(0, self.canvas_width, self.grid_space):
            self.canvas.create_line(0, i, self.canvas_width, i, fill='CadetBlue1')

    def create_grid(self):
        self.__vertical_lines__()
        self.__horizontal_lines__()

def createPyramid(apex, base, height):

    canvas_width = 400
    canvas_height = 400

    canvas = Canvas(root, width=canvas_width, height=canvas_height)
    canvas.grid(row=5, column=0, columnspan=2, sticky=E)
    canvas.configure(background='white')

    grid = GridLines(canvas, canvas_width, canvas_height, 20)
    GridLines.create_grid(grid)
    
    x_center = apex[0]
    y_top = apex[1]

    y_bottom = y_top + height
    y_middle = y_top + height / 1.6

    half_base = base / 2
    x_left = (x_center - (half_base))
    x_right = x_center + (half_base)
    
    right_offset = ((base * .6) - base)
    x_right_rear = x_right + right_offset

    left_offset = ((base * 1.1) - half_base)
    x_left_rear = x_center - left_offset
    
    # facing triangle
    points = [[x_left,y_bottom], [x_right,y_bottom], apex]
    canvas.create_polygon(points, outline='black', fill='Gray95')

    # left side shadow
    points3 = [apex, [x_left_rear,y_middle], [x_left,y_bottom]]
    canvas.create_polygon(points3, outline='black', fill='Gray85')

    # triangle lines
    canvas.create_line(x_center, y_top, x_right_rear, y_middle, fill='thistle3', dash=(4,4)) # back right
    canvas.create_line(x_right_rear, y_middle, x_left_rear, y_middle, fill='CadetBlue3', dash=(4,4)) # back middle
    canvas.create_line(x_right_rear, y_middle, x_left, y_bottom, fill='PaleGreen3', dash=(4,4)) # cross positive
    canvas.create_line(x_left_rear, y_middle, x_right, y_bottom, fill='PaleGreen3', dash=(4,4)) # cross negative
    canvas.create_line(x_right_rear, y_middle, x_right, y_bottom, fill='CadetBlue3', dash=(4,4)) # right connector

def clickFunction():

    apex = [200,100]
    pyramid_base = int(widthText.get())
    pyramid_height = int(heightText.get())
    
    # build instance of Pyramid
    new_pyramid = Pyramid(pyramid_base, pyramid_height)

    # display results of instance (outputs calculated data)
    pyramid_dimensions_output = Pyramid.createNewPyramid(new_pyramid)
    responseLabel = Label(root, text=pyramid_dimensions_output, justify=LEFT)
    responseLabel.grid(row=4, column=0, columnspan=2, ipadx='110', sticky=W)

    # outputs 3D graphic of pyramid (not to scale)
    createPyramid(apex, pyramid_base, pyramid_height)
    
root = Tk()

APP_NAME = 'Pyramid Builder'

root.iconbitmap('pyramid.ico')
root.title(APP_NAME)
root.geometry("600x700+1100+200")

header = Label(root, text=APP_NAME, font='Helvetica 18 bold')
header.grid(row=0, column=0, columnspan=2, pady='10')

# width entry
widthLabel = Label(root, text="Enter Base (in meters):")
widthLabel.grid(row=1, column=0, ipadx='30', sticky=W)
widthText = Entry(root)
widthText.grid(row=1, column=1, ipadx="100")
widthText.focus()

# height entry
heightLabel = Label(root, text="Enter Height (in meters):")
heightLabel.grid(row=2, column=0, ipadx='30', sticky=W)
heightText = Entry(root)
heightText.grid(row=2, column=1, ipadx="100")

# buttons
buttonFrame = Frame(root)
buttonFrame.grid(row=3, column=1, sticky=E)

submitButton = Button(buttonFrame, text="Submit", command=clickFunction)
closeButton = Button(buttonFrame, text='Close', command=root.destroy)

submitButton.pack(side='left', padx='2')
closeButton.pack(side='left', padx='2')

# root.grid_columnconfigure(0, minsize=80)
# root.grid_rowconfigure(0, pad=5)

root.mainloop()

# automated tests (would normally be in separate file just for testing)
print('\nrunning automated tests...')

import unittest
# would use "import Pyramid" here if in separate file

class TestPyramid(unittest.TestCase):
    # "self" is not normally required in: ClassName.methodCall(self), but needed because class is in same file as unit test
    
    def test_calculateBlockVolume(self):
        # only one test for block size (not supposed to change)
        self.assertEqual(Pyramid.calculateBlockVolume(self, 1.5, 2, 2.5), 7.5)

    def test_calculateGroundArea(self):
        self.assertEqual(Pyramid.calculateGroundArea(self, 80), 6400)
        self.assertEqual(Pyramid.calculateGroundArea(self, 236), 55696)

    def test_calculatePyramidVolume(self):
        self.assertEqual(Pyramid.calculatePyramidVolume(self, 6400, 64), 136533)
        self.assertEqual(Pyramid.calculatePyramidVolume(self, 55696, 138), 2562016)
        
    def test_countBlocks(self):
        self.assertEqual(Pyramid.countBlocks(self, 136533, 7.5), 18204)
        self.assertEqual(Pyramid.countBlocks(self, 2562016, 7.5), 341602)

    def test_calculateMass(self):
        self.assertEqual(Pyramid.calculateMass(self, 18204, 15000), 273060000)
        self.assertEqual(Pyramid.calculateMass(self, 341602, 15000), 5124030000)

# Runs the unit tests automatically in the current file
if __name__ == '__main__':
    unittest.main()

# ===========================================
# Visual of correct output for 2 test cases:
# results agree with automated unit testing
# and hand calculations (in MS Excel).
# ===================================
# input data: 
#   pyramid side length = 80 meters; 
#   pyramid height = 64 meters
# ========
# results: 
#   ground area covered = 6,400 m²
#   pyramid volume = 136,533 m³
#   blocks = 18,204
#   mass = 273,060,000 kg
# ===================================
# input data: 
#   pyramid side length = 236 meters; 
#   pyramid height = 138 meters
# ========
# results:
#   ground area covered = 55,696 m²
#   pyramid volume = 2,562,016 m³
#   blocks = 341,602
#   mass = 5,124,030,000 kg
# ===========================================
