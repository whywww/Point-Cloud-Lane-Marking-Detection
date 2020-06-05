import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

image = cv2.imread('points.jpg', cv2.IMREAD_GRAYSCALE)

threshold_high = 90
threshold_low = 20

image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

image = cv2.dilate(image, kernel=np.ones((4, 4)))

image = np.where(image < threshold_low, 0, image)
image = np.where(image > threshold_low, 255, image)

cv2.imwrite('result.png', image)