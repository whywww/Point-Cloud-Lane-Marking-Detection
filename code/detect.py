import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('../final_project_data/filtered_points.csv', delim_whitespace=True, usecols=['X','Y','Intensity'])
df_t = pd.read_csv('../final_project_data/trajectory.xyz', delim_whitespace=True, names=['X','Y','Z','Intensity'])

imin = df.min(axis=0)['Intensity']
imax = df.max(axis=0)['Intensity']

# print(df.max(axis=0))
# print(imin, imax)
xmax = int(df.max(axis=0)['X'])+1
ymax = int(df.max(axis=0)['Y'])+1


img = np.zeros((xmax,ymax))
img_t = np.zeros((xmax,ymax))

for i, row in df.iterrows():
    img[int(row['X']), int(row['Y'])] = float(row['Intensity'])/imax
for i, row in df_t.iterrows():
    img_t[int(row['X']), int(row['Y'])] = float(row['Intensity'])


# # edge detect
# img = (img * 255).astype(np.uint8)
# edges = cv2.Canny(img, 100, 200)

# edges = cv2.Canny(img,50,200,apertureSize = 3)

# Hough Transform
# minLineLength = 100
# maxLineGap = 5
# lines = cv2.HoughLines(edges,1,np.pi/180,5)
# for rho,theta in lines[0]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))

#     plt.plot((x1,y1),(x2,y2))

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.subplot(122),plt.imshow(img_t,cmap = 'gray', vmin=0, vmax=255)
plt.show()
