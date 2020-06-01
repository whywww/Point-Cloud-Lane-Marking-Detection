import pandas as pd 
from math import *

def load_fuse(data_path):
    mylist = []
    fh = open(data_path,'r')
    for i,line in enumerate(fh):
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
    a, rf = WGS84           # semi-major axis, reciprocal flattening
    e2 = 1 - (1 - 1 / rf) ** 2  # eccentricity squared
    n = a / sqrt(1 - e2 * sin_φ ** 2) # prime vertical radius
    r = (n + height) * cos(φ)   # perpendicular distance from z axis
    x = r * cos(λ)
    y = r * sin(λ)
    z = (n * (1 - e2) + height) * sin_φ
    return x, y, z

def convert_to_cartesian(df_probe):
    idx = 0
    min_lat = inf
    min_lon = inf
    min_alt = inf
    for lat, lon, alt in zip(df_probe['latitude'], df_probe['longitude'], df_probe['altitude']):
        lat = float(lat)
        lon = float(lon)
        alt = float(alt)
        x,y,z=converter(lat,lon,alt)
        df_probe.at[idx,'latitude'] = x
        df_probe.at[idx,'longitude'] = y
        df_probe.at[idx,'altitude'] = z
        min_lat = min(min_lat,x)
        min_lon = min(min_lon,y)
        min_alt = min(min_alt,z)
        idx+=1

    idx = 0
    for lat, lon, alt in zip(df_probe['latitude'], df_probe['longitude'], df_probe['altitude']):
        df_probe.at[idx,'latitude'] = lat-min_lat
        df_probe.at[idx,'longitude'] = lon-min_lon
        df_probe.at[idx,'altitude'] = alt - min_alt
        idx+=1
        # break
    return df_probe

if __name__ == "__main__":
    path = "../final_project_data/final_project_point_cloud.fuse"
    result = load_fuse(path)
    df_result = pd.DataFrame(result, columns = ['latitude','longitude','altitude','intensity']) 
    df_result = convert_to_cartesian(df_result)
    df_result.to_csv("../final_project_data/points", sep=" ", header=False, index=False)
    print(df_result)