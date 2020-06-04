import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('final_project_data/filtered_points.csv', delim_whitespace=True, usecols=['X','Y','Intensity'])

imin = df.min(axis=0)['Intensity']
imax = df.max(axis=0)['Intensity']
img_size = max(int(df.max(axis=0)['X']), int(df.max(axis=0)['Y'])) + 1

img = np.zeros((img_size*2,img_size*2))
for i, row in df.iterrows():
    img[int(row['X'])*2, int(row['Y'])*2] = float(row['Intensity'])/imax

# Trajectory
# df_t = pd.read_csv('final_project_data/trajectory.xyz', delim_whitespace=True, names=['X','Y','Z','Intensity'])
# img_t = np.zeros((xmax,ymax))
# for i, row in df_t.iterrows():
#     img_t[int(row['X']), int(row['Y'])] = float(row['Intensity'])

# edge detect
img = (img * 255).astype(np.uint8)
img = cv2.GaussianBlur(img, (7, 7), 0)
edges = cv2.Canny(img, 3, 20)

plt.subplot(133)
plt.axis((0,img_size*2,img_size*2,0))
plt.gca().set_aspect('equal', adjustable='box')

# Hough Transform
lines = cv2.HoughLines(edges,1,np.pi/180,40)
for i in range(0, len(lines)):
    for rho,theta in lines[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 100*(-b))
        y1 = int(y0 + 100*(a))
        x2 = int(x0 - 100*(-b))
        y2 = int(y0 - 100*(a))
        plt.plot((x1,y1),(x2,y2))
        # cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

plt.subplot(131),plt.imshow(img,cmap = 'gray')
plt.subplot(132),plt.imshow(edges,cmap = 'gray')
plt.show()