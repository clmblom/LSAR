import json
import numpy as np
import re


def read_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as data_file:
        data_dict = json.load(data_file)
        return data_dict


def interpolate_y(x_int, mid_coords):
    # x_int = [x_0, ..., x_n], x_0 < ... < x_n
    # mid_coords = [(x_0, y_0), ...., (x_m, y_m)], x_0 < ... < x_m
    mid_coords_np = np.array(mid_coords)
    y_int = np.interp(x_int, mid_coords_np[:, 0], mid_coords_np[:, 1]).astype(int)
    y_int = list(y_int)
    return y_int


def separate_string(s):
    if not s:
        return "", []
    class_pred = re.findall(r'<\w\w?\w?>', s)
    string_pred = re.sub(r'<\w\w?\w?>', '', s)
    return string_pred, class_pred