from typing import Tuple, Any
import math
import numpy as np
import open3d as o3d
import math
# import pymesh

def angle_with_x_axis(x, y):
    # 计算反正切值，注意参数顺序是 (y, x)
    angle_rad = np.arctan2(y, x)
    if y >= 0:
        if angle_rad < 0:
            angle_rad_adjusted = angle_rad + np.pi
        else:
            angle_rad_adjusted = angle_rad
    else:
        if angle_rad < 0:
            angle_rad_adjusted = angle_rad + np.pi + np.pi
        else:
            angle_rad_adjusted = angle_rad + np.pi
    return angle_rad_adjusted
class Torus:
    # 在文档和json中，theta是align_angle,phi是align_angle+end_angle，所以实例化类时start_angle=0,end_angle=phi-theta,align_angle=theta
    def __init__(self,torus_radius,radius,center_coord,start_angle,end_angle,normal,align_angle,id=-1,anchor_cylinders_idx=[]):
        self.torus_radius=torus_radius
        self.radius=radius # tube radius
        self.center_coord=center_coord
        self.normal=normal
        self.start_angle=start_angle
        self.end_angle=end_angle
        self.align_angle=align_angle
        self.anchor_cylinder=[]
        self.anchor_cylinders_idx=anchor_cylinders_idx
        self.id=id

    # 判断两个torus是否相同
    def __eq__(self, other):
        if isinstance(other, Torus):
            if np.allclose(self.torus_radius,other.torus_radius, rtol=1e-5) and np.allclose(self.radius,other.radius, rtol=1e-5)\
                and np.allclose(self.center_coord,other.center_coord, rtol=1e-5):
                return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def load_toruses_from_json(cls, json_file: str) -> 'list[Torus]':
        import json
        with open(json_file, 'r') as f:
            toruses = json.load(f)
        if isinstance(toruses, dict) and 'elbows' in toruses:
            toruses = toruses['elbows']
        assert isinstance(toruses, list)
        torus_list = []
        for torus in toruses:
            center = np.array(torus['center'])
            normal = np.array(torus['normal'])
            torus_list.append(cls(torus['radius1'], torus['radius2'], center, 0,
                                  float(torus['phi']) - float(torus['theta']), normal ,torus['theta'],torus['id'],torus['anchor_cylinders_idx']))
        return torus_list



    def is_point_near_the_surface_batch(self, points: 'np.ndarray[np.float32, (N, 3)]',
                                        threshold: float, sample_points: int = 100000) -> 'np.ndarray[np.bool8, (N)]':
        try:
            from util.nearest_neighbour import find_nearest_neighbors
        except ImportError:
            from util.nearest_neighbour import find_nearest_neighbors_stupid as find_nearest_neighbors

        o3d_mesh = self.to_o3d_mesh()
        sampled_points = o3d_mesh.sample_points_uniformly(sample_points)
        indices, distances = find_nearest_neighbors(points, np.asarray(sampled_points.points))
        return distances < threshold


    def get(self) -> tuple[Any, Any, Any, Any, Any, Any, list[Any]]:
        return self.center_coord,self.torus_radius, self.radius,self.normal,self.align_angle,self.align_angle+self.end_angle,self.anchor_cylinder

    def get_direction(self) -> 'np.array[float, float, float]':
        return self.normal / np.linalg.norm(self.normal)

    def get_rotation_matrix(self) -> 'np.array[[float, float, float], [float, float, float], [float, float, float]]':
        z = self.get_direction()
        y = np.cross(z, np.array([0, 0, 1]))
        y = y / np.linalg.norm(y) if np.abs(np.linalg.norm(y)) > 1e-6 else np.array([0, 1, 0])
        x = np.cross(y, z)
        assert np.abs(np.linalg.norm(x)) > 1e-6
        x = x / np.linalg.norm(x)
        rotation_matrix = np.array([x, y, z])
        return rotation_matrix


    def get_part_cylinder(self):
        cylinder = pymesh.generate_cylinder([0, 0, -self.radius], [0, 0, self.radius], self.torus_radius + self.radius,
                                            self.torus_radius + self.radius, 100)
        angle = [angle_with_x_axis(*x[:2]) for x in cylinder.vertices]
        angle = np.asarray(angle)
        vertex_indices = np.where(np.logical_and(angle > self.start_angle, angle < self.end_angle))
        vertex_indices = np.asarray(vertex_indices[0])
        select = [any([indice in vertex_indices for indice in face]) for face in cylinder.faces]
        select = np.where(select)[0]
        sub_noneplane = pymesh.submesh(cylinder, select, 0)

        vertices = np.array([
            [0.0, 0.0, self.radius],
            [0.0, 0.0, -self.radius],
            [np.cos(self.start_angle), np.sin(self.start_angle), -self.radius],
            [np.cos(self.start_angle), np.sin(self.start_angle), self.radius],
            [np.cos(self.end_angle), np.sin(self.end_angle), self.radius],
            [np.cos(self.end_angle), np.sin(self.end_angle), -self.radius]
        ])
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 4, 5],
            [0, 5, 1]
        ])
        plane = pymesh.form_mesh(vertices, faces)
        part_cylinder = pymesh.boolean(sub_noneplane, plane, operation='union')
        return plane, part_cylinder

    # 该函数修改后未进行测试
    def get_boolean(self):
        #存在外径小于内径的情况
        full_torus = o3d.geometry.TriangleMesh.create_torus(torus_radius=self.torus_radius,
                                                            tube_radius=self.radius,
                                                            radial_resolution=80,
                                                            tubular_resolution=60)
        vertices = np.asarray(full_torus.vertices)
        triangles = np.asarray(full_torus.triangles)
        py_torus=pymesh.form_mesh(vertices,triangles)
        # pymesh.save_mesh("pymeshobj/test.obj",py_torus)
        plane,part_cylinder=self.get_part_cylinder()
        # pymesh.save_mesh("pymeshobj/part_cylinder.obj",part_cylinder)
        # pymesh.save_mesh("pymeshobj/py_torus.obj",py_torus)
        final_mesh=pymesh.boolean(py_torus,part_cylinder,operation='union')
        pymesh.save_mesh("pymeshobj/final_mesh.obj",final_mesh)
        vertices = final_mesh.vertices
        vertices=np.asarray(vertices).copy()
        faces = final_mesh.faces
        faces=np.asarray(faces).copy()
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
        mesh_o3d.compute_vertex_normals()
        return mesh_o3d

    def get_partial(self) -> 'o3d.geometry.TriangleMesh':
        full_torus=o3d.geometry.TriangleMesh.create_torus(torus_radius=self.torus_radius,tube_radius=self.radius,
                                                          radial_resolution=80,
                                                          tubular_resolution=60)
        vertices = np.asarray(full_torus.vertices)
        triangles = np.asarray(full_torus.triangles)
        angles = np.arctan2(vertices[:, 1], vertices[:, 0])
        # 这里设置误差阈值，因为可能在角度边界处因为顶点越界无法获得三角网格
        condition = np.logical_and(self.start_angle - 0.05 * np.pi <= angles, angles <= self.end_angle + 0.05 * np.pi)
        selected_indices = np.where(condition)[0]

        selected_triangles = []
        for triangle in triangles:
            if all(index in selected_indices for index in triangle):
                selected_triangles.append(triangle)

        partial_torus = o3d.geometry.TriangleMesh()
        partial_torus.vertices = o3d.utility.Vector3dVector(vertices)
        partial_torus.triangles = o3d.utility.Vector3iVector(selected_triangles)
        # sampled_points = partial_torus.sample_points_uniformly(10000)
        #
        # o3d.visualization.draw_geometries([partial_torus,sampled_points])
        return partial_torus

    def to_o3d_mesh(self) -> 'o3d.geometry.TriangleMesh':
        # 水密
        # partial_torus=self.get_boolean()
        # 不需要水密时，为了计算效率采用下面一行代码
        partial_torus=self.get_partial()

        local_rotation_matrix = np.array([[np.cos(self.align_angle), np.sin(self.align_angle), 0],
                                          [-np.sin(self.align_angle), np.cos(self.align_angle), 0],
                                          [0, 0, 1]])  # 绕 z 轴旋转
        partial_torus.rotate(local_rotation_matrix, center=(0, 0, 0))
        self.partial_torus = partial_torus

        rotation_matrix=self.get_rotation_matrix()
        self.partial_torus.rotate(rotation_matrix.T,center=(0,0,0))
        self.partial_torus.translate(self.center_coord)
        self.partial_torus.compute_vertex_normals()
        return self.partial_torus