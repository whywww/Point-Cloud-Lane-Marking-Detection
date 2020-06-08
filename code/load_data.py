import utm
import pandas as pd
from math import *


def load_fuse(data_path):
    mylist = []
    fh = open(data_path, 'r')
    for i, line in enumerate(fh):
        lines = []
        for cell in line.split():
            lines.append(cell)
        mylist.append(lines)
    fh.close()
    return mylist


def converter(latitude, longitude, height):
    """Return geocentric (Cartesian) Coordinates x, y, z corresponding to
    the geodetic coordinates given by latitude and longitude (in
    degrees) and height above ellipsoid. The ellipsoid must be
    specified by a pair (semi-major axis, reciprocal flattening).

    """
    WGS84 = 6378137, 298.257223563
    φ = radians(latitude)
    λ = radians(longitude)
    sin_φ = sin(φ)
    a, rf = WGS84  # semi-major axis, reciprocal flattening
    e2 = 1 - (1 - 1 / rf) ** 2  # eccentricity squared
    n = a / sqrt(1 - e2 * sin_φ ** 2)  # prime vertical radius
    r = (n + height) * cos(φ)  # perpendicular distance from z axis
    x = r * cos(λ)
    y = r * sin(λ)
    z = (n * (1 - e2) + height) * sin_φ
    return x, y, z


def convert_to_cartesian(df_probe):
    idx = 0
    # min_lat = inf
    # min_lon = inf
    # min_alt = inf
    min_x = 657227.2732683367
    min_y = 5085303.877106553
    for lat, lon, alt in zip(df_probe['latitude'], df_probe['longitude'], df_probe['altitude']):
        lat = float(lat)
        lon = float(lon)
        alt = float(alt)
        new_coordinate = utm.from_latlon(lat, lon)
        x = new_coordinate[0]
        y = new_coordinate[1]
        z = alt
        # min_x = min(min_x,x)
        # min_y = min(min_y,y)
        # min_z = min(min_z,z)
        df_probe.at[idx, 'latitude'] = x - min_x
        df_probe.at[idx, 'longitude'] = y - min_y
        df_probe.at[idx, 'altitude'] = z
        idx += 1  
    # print("min_x: ", min_x, " min_y: ", min_y, " min_z: ",min_z)
    # idx=0
    # for lat, lon, alt in zip(df_probe['latitude'], df_probe['longitude'], df_probe['altitude']):
    #     df_probe.at[idx, 'latitude'] = lat-min_x
    #     df_probe.at[idx, 'longitude'] = lon-min_y
    #     # df_probe.at[idx, 'altitude'] = alt - min_z
    #     idx += 1  
    return df_probe

if __name__ == "__main__":
    # path = "../final_project_data/final_project_point_cloud.fuse"
    # result = load_fuse(path)
    # df_result = pd.DataFrame(result, columns=['latitude', 'longitude', 'altitude', 'intensity'])
    # df_result = convert_to_cartesian(df_result)
    # df_result.to_csv("../results/cloudpoints.xyz", sep=" ", header=False, index=False)
    # print(df_result)

    path = "../final_project_data/trajectory.fuse"
    result = load_fuse(path)
    df_result = pd.DataFrame(result, columns=['latitude', 'longitude', 'altitude', 'intensity'])
    df_result = convert_to_cartesian(df_result)
    df_result.to_csv("../results/trajectory.xyz", sep=" ", header=False, index=False)
    print(df_result)

 