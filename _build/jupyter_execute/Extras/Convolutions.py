# Convolution Filter Example

import numpy as np 
import cv2
import matplotlib.pyplot as plt
from pylab import rcParams
from matplotlib import gridspec

sharpen = np.array([[0, -1, 0], 
                   [-1, 5, -1], 
                   [0, -1, 0]])

laplacian = np.array([[0, 1, 0], 
                      [1, -4, 1], 
                      [0, 1, 0]])

emboss = np.array([[-2, -1, 0], 
                   [-1, 1, 1], 
                   [0, 1, 2]])

outline = np.array([[-1, -1, -1], 
                    [-1, 8, -1], 
                    [-1, -1, -1]])

bottom_sobel = np.array([[-1, -2, -1], 
                         [0, 0, 0], 
                         [1, 2, 1]])

left_sobel = np.array([[1, 0, -1], 
                       [2, 0, -2], 
                       [1, 0, -1]])

right_sobel = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]])

top_sobel = np.array([[1, 2, 1], 
                      [0, 0, 0], 
                      [-1, -2, -1]])

filters = (sharpen, laplacian, emboss, outline, bottom_sobel, left_sobel, right_sobel, top_sobel)
filterstring = ('sharpen', 'laplacian', 'emboss', 'outline', 'bottom_sobel', 'left_sobel', 'right_sobel', 'top_sobel')

rcParams['figure.figsize'] = 16, 15
rcParams.update({'font.size': 12})

image = cv2.imread('C://Users//Samuel//Documents//samph4//TrainingBook//Examples/Figures/cow.png')
image_rgb = image.copy()
image_rgb[:, :, 0] = image[:, :, 2]
image_rgb[:, :, 2] = image[:, :, 0]


# MNIST test input (1st subplot)
plt.imshow(image_rgb)
plt.axis('off')
plt.title('original image')
plt.show()



# MNIST test input (1st subplot)
for i in range(8):
    plt.subplot(4,2,i+1)
    result = cv2.filter2D(image_rgb, -1, filters[i])
    plt.imshow(result)
    plt.axis('off')
    plt.title(filterstring[i])

plt.show()

