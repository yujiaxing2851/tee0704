import numpy as np
import open3d as o3d
import json
from cylinder import Cylinder
from torus import Torus

TOP = 0
BOTTOM = 1


# 自定义编码器类
class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.ndarray):
            return list(obj)  # 将 ndarray 转换为 Python 列表
        else:
            return super().default(obj)


def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def get_rotation_from_cy1_to_cy2(cylinder, cylinder_side, new_cylinder):
    """
      计算将向量v1旋转到向量v2的旋转矩阵。

      参数:
      v1 (array-like): 初始向量
      v2 (array-like): 目标向量

      返回:
      numpy.ndarray: 旋转矩阵
    """
    v1 = get_correct_dir(cylinder, cylinder_side)
    v2 = get_correct_dir(new_cylinder, cylinder_side)
    # 计算旋转轴（叉积）
    axis = np.cross(v1, v2)
    axis_length = np.linalg.norm(axis)

    # 计算旋转角度（点积和反余弦）
    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

    # 处理特殊情况：v1 和 v2 平行或反平行
    if axis_length == 0:
        if np.dot(v1, v2) > 0:
            return np.identity(3)  # v1 和 v2 平行，返回单位矩阵
        else:
            return -np.identity(3)  # v1 和 v2 反平行，返回负单位矩阵

    axis = axis / axis_length

    # 使用 Rodrigues' 旋转公式生成旋转矩阵
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])

    R = np.identity(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    return R


def get_coord_error(new_cylinder, cylinder):
    coord1 = (new_cylinder.top_center + new_cylinder.bottom_center) / 2
    coord2 = (cylinder.top_center + cylinder.bottom_center) / 2
    return coord1 - coord2


def neighbor_elbow(coord1, coord2, gap):
    return np.linalg.norm(np.array(coord1) - np.array(coord2)) < gap


def get_neighbors(top_bottom, given_vector):
    # 存储距离和对应下标列表
    distances = []
    for i, tup in enumerate(top_bottom):
        for j, vec in enumerate(tup):
            distance = euclidean_distance(given_vector, vec)
            distances.append((distance, (i, j)))
    distances.sort(key=lambda x: x[0])
    sorted_indices = [index for _, index in distances]
    return sorted_indices


# 获取指向所选面外侧的法向量
def get_correct_dir(cylinder, side):
    if side == TOP:
        delta = np.array(cylinder.top_center) - np.array(cylinder.bottom_center)
    else:
        delta = np.array(cylinder.bottom_center) - np.array(cylinder.top_center)
    return delta / np.linalg.norm(delta)


def save_json(elbows_for_para):
    json_cylinders = []
    json_elbows = []
    json_cylinder_idx = 0
    json_elbow_idx = 0

    # 保存json elbows是Elbow的list
    for instance in elbows_for_para:
        if isinstance(instance, Cylinder):
            json_results = instance.get()
            json_data = {
                "top_center": json_results[0],
                "bottom_center": json_results[1],
                "radius": json_results[2],
                "id": json_cylinder_idx
            }
            json_cylinder_idx += 1
            # json_string = json.dumps(json_data, cls=EnumEncoder)
            json_cylinders.append(json_data)
        elif isinstance(instance, Torus):  # torus instance
            json_results = instance.get()
            json_data = {
                "center": json_results[0],
                "radius1": json_results[1],
                "radius2": json_results[2],
                "normal": json_results[3],
                "theta": json_results[4],
                "phi": json_results[5],
                "id": json_elbow_idx
            }
            json_elbow_idx += 1
            # json_string = json.dumps(json_data, cls=EnumEncoder)
            json_elbows.append(json_data)

    json_final = []
    json_final.append({"cylinders": json_cylinders})
    json_final.append({"elbows": json_elbows})
    json_final.append({"tees": {}})
    json_final.append({"crosses": {}})
    json_string = json.dumps({"cylinders": json_cylinders, "elbows": json_elbows, "tees": [], "crosses": []},
                             cls=EnumEncoder)
    with open("./parameters_0704.json", 'a') as outfile:
        outfile.write(json_string)
