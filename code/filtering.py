import numpy as np
import pandas as pd


def load_file(path):
    data = []

    with open(path) as fuse:
        line = fuse.readline()
        # print(line)
        while line:
            # for _ in range(100000):
            data.append([float(x) for x in line.split()])
            line = fuse.readline()

    return np.array(data)


def filter1(pts):
    # filter the data with intensity

    intensity = pts[:, 3]
    mean = np.mean(intensity)
    std = np.std(intensity)

    threshold = mean + std  # adjust this value to see different results
    print(f'threshold is {threshold}')

    filtered_points = []

    for point in pts:
        if point[3] > threshold:
            filtered_points.append(point)
    print(f'after filter 1 shape {np.array(filtered_points).shape}')
    return np.array(filtered_points)


def filter2(pts):
    with open('../final_project_data/trajectory.xyz') as tra:
        data = tra.readline()
        trajectory = []
        while data:
            trajectory.append([float(x) for x in data.split()])
            data = tra.readline()

    trajectory = np.array(trajectory)[:, :2]
    # print(f'trajectory is {trajectory}')

    threshold = 20  # in meters. adjust this to see different results

    result = []
    for point in pts:
        dist = min(np.sum((trajectory - point[:2]) ** 2, axis=1)) ** 0.5
        if dist < threshold:
            result.append(point)
    print(f'after filter 2 shape {np.array(result).shape}')
    return np.array(result)


def filter3(pts):

    # filter with elevation

    result = []

    mean = np.mean(pts[:, 2])
    std = np.std(pts[:, 2])

    threshold = mean * 1.5

    for point in pts:
        if point[2] < threshold:
            result.append(point)

    print(f'after filter 3 shape {np.array(result).shape}')
    return np.array(result)


if __name__ == '__main__':
    points = load_file('../final_project_data/cloudpoints.xyz')

    points = filter1(points)
    points = filter2(points)
    points = filter3(points)

    result = pd.DataFrame(points)

    result.to_csv("../final_project_data/filtered_points.csv", sep=" ", header=['X', 'Y', 'Z', 'Intensity'],
                  index=False)
