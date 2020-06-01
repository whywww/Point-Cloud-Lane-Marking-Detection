import numpy as np
import pandas as pd


def load_file(path):
    data = []

    with open(path) as fuse:
        line = fuse.readline()

        for _ in range(1000):  # load the first 1000 point for testing

            data.append([float(x) for x in line.split()])
            line = fuse.readline()

    return np.array(data)


def filter1(pts):
    # filter the data with intensity

    intensity = pts[:, 3]
    mean = np.mean(intensity)
    std = np.std(intensity)

    threshold = mean + std  # adjust this value to see different results

    filtered_points = []

    for point in pts:
        if point[3] > threshold:
            filtered_points.append(point)

    return np.array(filtered_points)


def filter2(pts):

    with open('../final_project_data/trajectory.fuse') as tra:
        data = tra.readline()
        trajectory = []
        while data:
            trajectory.append([float(x) for x in data.split()])
            data = tra.readline()

    trajectory = np.array(trajectory)[:, :3]

    threshold = 20  # in meters. adjust this to see different results
    result = []
    for point in pts:
        dist = min(np.sum((trajectory-point[:3])**2, axis=1))
        if dist < threshold:
            result.append(point)

    return np.array(result)


if __name__ == '__main__':
    points = load_file('../final_project_data/final_project_point_cloud.fuse')

    points = filter1(points)
    points = filter2(points)
