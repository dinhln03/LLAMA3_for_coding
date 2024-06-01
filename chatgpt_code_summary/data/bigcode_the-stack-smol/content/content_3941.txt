import cv2
import matplotlib.pyplot as plt
import numpy as np
img= cv2.imread("img.png")
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.axis('off')
# show the image
plt.imshow(img)
plt.show()

# get the image shape
rows, cols, dim = img.shape
rows, cols, dim = img.shape
# transformation matrix for Shearing
# shearing applied to x-axis
M1 = np.float32([[1, 0.5, 0],
             	[0, 1  , 0],
            	[0, 0  , 1]])
# shearing applied to y-axis
M2 = np.float32([[1,   0, 0],
            	  [0.5, 1, 0],
            	  [0,   0, 1]])
# apply a perspective transformation to the image
sheared_img_in_x = cv2.warpPerspective(img,M1,(int(cols*1.5),int(rows*1.5)))
sheared_img_in_y = cv2.warpPerspective(img,M2,(int(cols*1.5),int(rows*1.5)))
# disable x & y axis
plt.axis('off')
# show the resulting image
plt.subplot(121)
plt.imshow(sheared_img_in_x)
plt.subplot(122)
plt.imshow(sheared_img_in_y)
plt.show()