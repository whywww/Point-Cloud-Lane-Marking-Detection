import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

img = cv2.imread('../points.jpg', 0)
img_size = img.shape[0]

# edge detect
img = cv2.GaussianBlur(img, (7, 7), 0)
img = cv2.medianBlur(img, 3)
edges = cv2.Canny(img, 10, 20)

plt.subplot(133)
plt.axis((0, img_size, img_size, 0))
plt.gca().set_aspect('equal', adjustable='box')

# Hough Transform
threshold = 50
minLineLength = 50
maxLineGap = 5
width = 2  # width of line marking
avg_k = 0
set_b = []
try:
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength, maxLineGap)
    for i in range(0, len(lines)):
        # for rho,theta in lines[i]:
        #     a = np.cos(theta)
        #     b = np.sin(theta)
        #     x0 = a*rho
        #     y0 = b*rho
        #     x1 = int(x0 - 2000*(-b))
        #     y1 = int(y0 - 2000*(a))
        #     x2 = int(x0 - 500*(-b))
        #     y2 = int(y0 - 500*(a))
        for x1, y1, x2, y2 in lines[i]:
            k = (y2 - y1) / (x2 - x1)
            b = y1 - k * x1
            avg_k += k
            plt.plot((x1, x2), (y1, y2))
            # print(b)

except:
    print('No lines detected')
    pass

avg_k /= len(lines)
print(avg_k)

# Clustering
# db = DBSCAN(eps=0.3, min_samples=10).fit(img)
# labels = db.labels_
# print(np.reshape(labels, [img_size, img_size]))

plt.subplot(131), plt.imshow(img, cmap='gray')
plt.subplot(132), plt.imshow(edges, cmap='gray')
plt.show()
