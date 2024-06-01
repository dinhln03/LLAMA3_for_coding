import numpy as np
import cv2 as cv
import math
from server.cv_utils import * 

def filterGaussian(img,size=(5,5),stdv=0):
    """Summary of filterGaussian
    This will apply a noise reduction filter, we will use s 5x5 Gaussian filter to smooth
        the image to lower the sensitivity to noise. (The smaller the size the less visible the blur)

    To populate the Gaussian matrix we will use a kernel of normally distributed[stdv=1] numbers which will
        set each pixel value equal to the weighted average of its neighboor pixels

    The Gaussian distribution:
        Gd = (1/2pi*stdv^2)exp(-((i-(k+1)^2) + (j - (k+1)^2))/(2*stdv^2))

        i,j E [1,2k+1] for the kernel of size: (2k+1)x(2k+1) 
    """

    if not isCV(img):
        raise ValueError("Image not in np.array format")

    if not isinstance(size,tuple):
        raise ValueError('filterGaussian: Size for Gaussian filter not tuple')
    return cv.GaussianBlur(img,size,stdv)


def filterCanny(img,min_val=50,max_val=150,size=(5,5),stdv=0):
    """
    The Canny detector is a multi-stage algorithm optimized for fast real-time edge detection, 
        which will reduce complexity of the image much further.

    The algorithm will detect sharp changes in luminosity and will define them as edges.

    The algorithm has the following stages:
        -   Noise reduction
        -   Intensity gradient - here it will apply a Sobel filter along the x and y axis to detect if edges are horizontal vertical or diagonal
        -   Non-maximum suppression - this shortens the frequency bandwith of the signal to sharpen it
        -   Hysteresis thresholding
    """
    if not isCV(img):
        raise ValueError("Image not in np.array format")
    
    if min_val >= max_val:
        raise ValueError('filterCanny: Value order incorrect')
    
    gray_scale = toGrayScale(img)
    #cv.imshow('Gray Scale image',gray_scale)

    gaussian = filterGaussian(gray_scale,size=size,stdv=stdv)
    #cv.imshow('Gaussian filter',gaussian)
    return cv.Canny(gaussian,min_val,max_val)


def segmentRegionOfInterest(img):
    height = img.shape[0]

    polygons = np.array([ 
        [(200, height), (1100, height), (550, 250)] 
        ]) 
    mask = np.zeros_like(img) 

    # Fill poly-function deals with multiple polygon 
    cv.fillPoly(mask, polygons, 255)

    # Bitwise operation between canny image and mask image 
    masked_image = cv.bitwise_and(img, mask)  
    return masked_image 


def houghFilter(frame,distance_resolution=2,angle_resolution=np.pi/180,min_n_intersections=50,min_line_size=30,max_line_gap=5):
    """
    Params:
        frame
        distance_resolution:    distance resolution of accumulator in pixels, larger ==> less precision
        angle_resolution:   angle of accumulator in radians, larger ==> less precision
        min_n_intersections: minimum number of intersections
        min_line_size:  minimum length of line in pixels
        max_line_gap:   maximum distance in pixels between disconnected lines
    """
    placeholder = np.array([])
    hough = cv.HoughLinesP(frame,distance_resolution,angle_resolution,min_n_intersections,placeholder,min_line_size,max_line_gap)
    return hough


def calculateLines(img,lines):
    """
    Combines line segments into one or two lanes
    Note: By looking at the slop of a line we can see if it is on the left side (m<0) or right (m>0)
    """
    def calculateCoordinates(img,line_params):
        """
        Calculates the coordinates for a road lane
        """
        #y = m*x +b, m= slope, b=intercept
        height, width, _ = img.shape
        m, b = line_params  
        y1 = height
        y2 = int(y1 * (1/2)) # make points from middle of the frame down
        
        # bound the coordinates within the frame
        x1 = max(-width, min(2 * width, int((y1 - b) / m)))
        x2 = max(-width, min(2 * width, int((y2 - b) / m)))

        return np.array([x1,y1, x2,y2])

    lane_lines = []
    if lines is None:
        return np.array(lane_lines)
    
    height, width, _ = img.shape
    left_lines, right_lines = [], []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary # right lane line segment should be on left 2/3 of the screen


    for line in lines:
        x1,y1, x2,y2 = line.reshape(4)

        if x1 == x2:
            #Vertical line
            continue

        #Fit a polynomial to the points to get the slope and intercept
        line_params = np.polyfit((x1,x2), (y1,y2), 1)
        slope,intercept = line_params[0], line_params[1]

        if slope < 0: #left side
            if x1 < left_region_boundary and x2 < left_region_boundary:
                left_lines.append((slope,intercept))
        else: #right
            if x1 > right_region_boundary and x2 > right_region_boundary:
                right_lines.append((slope,intercept))

    left_lines_avg = np.average(left_lines,axis=0)
    right_lines_avg = np.average(right_lines,axis=0)

    if len(left_lines) > 0:
        left_line = calculateCoordinates(img,left_lines_avg)
        lane_lines.append(left_line)
    
    if len(right_lines) > 0:
        right_line = calculateCoordinates(img,right_lines_avg)
        lane_lines.append(right_line)

    return np.array(lane_lines)


def showMidLine(img,steering_angle,color=(0, 255, 0),thickness=5):
    line_image = np.zeros_like(img)
    height, width, _ = img.shape

    # Note: the steering angle of:
    # 0-89 degree: turn left
    # 90 degree: going straight
    # 91-180 degree: turn right 

    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv.line(line_image, (x1, y1), (x2, y2), color, thickness)

    return line_image


def showLines(img,lines,color=(255,0,0),thickness=5):
    line_img = np.zeros(img.shape, dtype=np.uint8)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv.line(line_img, (x1,y1), (x2,y2), color, thickness)
    return line_img
    

def calculateSteeringAngle(img,lines):
    if len(lines) == 0:
        return -90
    
    height, width, _ = img.shape
    if len(lines) == 1:
        x1, _, x2, _ = lines[0]
        x_offset = x2 - x1
    else: #2 lines
        _, _, left_x2, _ = lines[0]
        _, _, right_x2, _ = lines[1]
        camera_mid_offset_percent = 0.0 # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
        mid = int(width / 2 * (1 + camera_mid_offset_percent))
        x_offset = (left_x2 + right_x2) / 2 - mid

    # find the steering angle, which is angle between navigation direction to end of center line
    y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
    steering_angle = angle_to_mid_deg + 90  # this is the steering angle needed by picar front wheel

    return steering_angle


def stabilizeSteeringAngle(curr_steering_angle, new_steering_angle, num_of_lane_lines, max_angle_deviation_two_lines=2, max_angle_deviation_one_lane=1):
    """
    Using last steering angle to stabilize the steering angle
    This can be improved to use last N angles, etc
    if new angle is too different from current angle, only turn by max_angle_deviation degrees
    """

    if num_of_lane_lines == 1:
        # if only one lane detected, don't deviate too much
        max_angle_deviation = max_angle_deviation_one_lane
    else:
        # if both lane lines detected, then we can deviate more
        max_angle_deviation = max_angle_deviation_two_lines
    
    angle_deviation = new_steering_angle - curr_steering_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering_angle = int(curr_steering_angle
                                        + max_angle_deviation * angle_deviation / abs(angle_deviation))
    else:
        stabilized_steering_angle = new_steering_angle

    return stabilized_steering_angle


