import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Read Image
img = cv2.imread('result.png', 0)
img_size = img.shape[0]
print('Image Shape is: ', img.shape)

"""
    edge detect
"""
img = cv2.GaussianBlur(img, (7, 7), 0)
img = cv2.medianBlur(img, 3)
edges = cv2.Canny(img, 10, 20)
plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('Filtered Point Cloud')
plt.subplot(222), plt.imshow(edges, cmap='gray'), plt.title('Edge Detection')


"""
    Hough Transform
""" 
plt.subplot(223)
plt.title('Hough Line Transform')
plt.axis((0, img_size, img_size, 0))
plt.gca().set_aspect('equal', adjustable='box')

threshold = 50
minLineLength = 50
maxLineGap = 10

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
            plt.plot((x1,x2),(y1,y2))
            # cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
except:
    print('No lines detected')
    pass


"""
    Clustering
""" 
plt.subplot(224)
plt.title('DBSCAN Clustering')
# plt.axis((0, img_size, img_size, 0))
plt.gca().set_aspect('equal', adjustable='box')

lane_points_ind = np.argwhere(edges)
X = StandardScaler().fit_transform(lane_points_ind)

db = DBSCAN(eps=0.07, min_samples=20).fit(X)

# X = (X - np.amin(X)) / (np.amax(X) - np.amin(X)) * img_size

labels = db.labels_
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print(n_clusters_)
print(n_noise_)

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], markerfacecolor=tuple(col))

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1],markerfacecolor=tuple(col))

plt.show()