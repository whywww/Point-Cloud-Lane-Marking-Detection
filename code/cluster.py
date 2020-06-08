import pandas as pd
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


df = pd.read_csv('results/filtered_points.csv', delim_whitespace=True, usecols=['X','Y'])
X = StandardScaler().fit_transform(df)
db = DBSCAN(eps=0.1, min_samples=20).fit(X)
labels = db.labels_
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print(n_clusters_)
print(n_noise_)


width = 4  # width of line markings
clusters = []

# Visualize
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        continue

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    # print(xy.shape)
    plt.scatter(xy[:, 0], xy[:, 1], s=1)  # c=tuple(col),

    # two ends
    x1, y1 = np.amax(xy,axis=0)
    x2, y2 = np.amin(xy,axis=0)

    k = (y2-y1) / (x2-x1)
    b = y1 - k*x1
    clusters.append((k, b, x1, x2))

# print(clusters)
plt.show()


# find all points
# lanes = []
# df_cloud = pd.read_csv('results/more_filtered_points.csv', delim_whitespace=True, usecols=['X','Y','Z','Intensity'], nrows=100)
# for i, row in df_cloud.iterrows():
#     for k, b, x1, x2 in clusters:
#         if (row['Y'] >= k * row['X'] + b - width/2) and (row['Y'] <= k * row['X'] + b + width/2) and (row['X'] <= x1) and (row['X'] >= x2):
#             lanes.append([row['X'], row['Y'], row['Z'], row['Intensity']])

# df_lanes = pd.DataFrame(np.array(lanes))
# df_lanes.to_csv("results/result.csv", sep=" ", header=['X', 'Y', 'Z', 'Intensity'], index=False)