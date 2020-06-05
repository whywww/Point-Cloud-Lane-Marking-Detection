import pandas as pd
import cv2
import numpy as np


df = pd.read_csv('final_project_data/filtered_points.csv', delim_whitespace=True, usecols=['X','Y','Intensity'])
# df_t = pd.read_csv('final_project_data/trajectory.xyz', delim_whitespace=True, names=['X','Y','Z','Intensity'])

factor = 8

imax = df.max(axis=0)['Intensity']
img_size = max(int(df.max(axis=0)['X']), int(df.max(axis=0)['Y'])) + 1

img = np.zeros((img_size*factor,img_size*factor))
for i, row in df.iterrows():
    img[int(row['X']*factor), int(row['Y']*factor)] = float(row['Intensity'])/imax
img = (img * 255).astype(np.uint8)

cv2.imwrite('points.jpg', img)

# Trajectory
# img_t = np.zeros((xmax,ymax))
# for i, row in df_t.iterrows():
#     img_t[int(row['X']), int(row['Y'])] = float(row['Intensity'])