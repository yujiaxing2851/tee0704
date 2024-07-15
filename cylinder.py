import numpy as np
import open3d as o3d
import math
class Cylinder:
    def __init__(self, top_center: 'tuple[float, float, float]', bottom_center: 'tuple[float, float, float]', radius: float,id=-1,anchor_torus_idx=[]):
        self.top_center = top_center
        self.bottom_center = bottom_center
        self.radius = radius
        self.points = None
        self.anchor_torus=[]
        self.anchor_torus_idx=anchor_torus_idx
        self.id=id

    # 判断两个圆柱是否相同
    def __eq__(self, other):
        if isinstance(other,Cylinder):
            if np.allclose(self.top_center,other.top_center, rtol=1e-5) and np.allclose(self.bottom_center,other.bottom_center, rtol=1e-5)\
                and np.allclose(self.radius,other.radius, rtol=1e-5):
                return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def get(self) -> 'tuple[tuple[float, float, float], tuple[float, float, float], float]':
        return self.top_center, self.bottom_center, self.radius,self.anchor_torus

    def split_get(self) -> 'tuple[float, float, float, float, float, float, float]':
        return self.top_center[0], self.top_center[1], self.top_center[2], self.bottom_center[0], \
            self.bottom_center[1], self.bottom_center[2], self.radius

    def numpy_get(self) -> 'np.ndarray[np.float32, (7, 1)]':
        return np.array(
            [self.top_center[0], self.top_center[1], self.top_center[2], self.bottom_center[0], self.bottom_center[1],
             self.bottom_center[2], self.radius])

    def get_direction(self) -> 'np.ndarray[np.float32, (3, 1)]':
        delta = np.array(self.top_center) - np.array(self.bottom_center)
        return delta / np.linalg.norm(delta)

    def get_height(self) -> float:
        return np.linalg.norm(np.array(self.top_center) - np.array(self.bottom_center))

    def is_point_inside(self, point: 'np.ndarray[np.float32, (3, 1)]') -> bool:
        direction = self.get_direction()
        delta = np.array(point) - np.array(self.bottom_center)
        if np.linalg.norm(np.cross(delta, direction)) > self.radius:
            return False
        if np.dot(delta, direction) < 0 or np.dot(delta, direction) > np.linalg.norm(
                np.array(self.bottom_center) - np.array(self.top_center)):
            return False
        return True

    def is_point_near_the_surface(self, point: 'np.ndarray[np.float32, (3, 1)]', threshold: float) -> bool:
        direction = self.get_direction()
        delta = np.array(point) - np.array(self.bottom_center)
        if np.linalg.norm(np.cross(delta, direction)) > self.radius + threshold or \
                np.linalg.norm(np.cross(delta, direction)) < self.radius - threshold:
            return False
        if np.dot(delta, direction) < 0 - threshold or np.dot(delta, direction) > self.get_height() + threshold:
            return False
        return True

    def is_point_near_the_surface_batch(self, points: 'np.ndarray[np.float32, (N, 3)]',
                                        threshold: float) -> 'np.ndarray[np.bool_, (N)]':
        """
        判断点云中每个点距离圆柱距离是否小于 threshold
        """

        direction = self.get_direction()
        delta = np.array(points) - np.array(self.bottom_center)
        delta_cross = np.linalg.norm(np.cross(delta, direction), axis=1)
        delta_dot = np.dot(delta, direction)
        condition1 = np.logical_and(delta_cross > self.radius - threshold,
                                    delta_cross < self.radius + threshold)
        condition2 = np.logical_and(delta_dot > -threshold,
                                    delta_dot < self.get_height() + threshold)
        return np.logical_and(condition1, condition2)

    def is_point_near_the_surface_batch_stupid(self, points: 'np.ndarray[np.float32, (N, 3)]', threshold: float,
                                               sample_points: int = 100000) -> 'np.ndarray[np.bool_, (N)]':
        try:
            from util.nearest_neighbour import find_nearest_neighbors
        except ImportError:
            from util.nearest_neighbour import find_nearest_neighbors_stupid as find_nearest_neighbors

        o3d_mesh = self.to_o3d_mesh()
        sampled_points = o3d_mesh.sample_points_uniformly(sample_points)
        indices, distances = find_nearest_neighbors(points, np.asarray(sampled_points.points))
        return distances < threshold

    def get_rotation_matrix(self) -> 'np.ndarray[np.float32, (3, 3)]':
        z = self.get_direction()
        y = np.cross(z, np.array([0, 0, 1]))
        y = y / np.linalg.norm(y) if np.abs(np.linalg.norm(y)) > 1e-6 else np.array([0, 1, 0])
        x = np.cross(y, z)
        assert np.abs(np.linalg.norm(x)) > 1e-6
        x = x / np.linalg.norm(x)
        rotation_matrix = np.array([x, y, z])
        return rotation_matrix

    def to_o3d_mesh(self) -> 'o3d.geometry.TriangleMesh':
        cylinder_mesh = o3d.geometry.TriangleMesh.create_cylinder(self.radius, self.get_height())
        rotation_matrix = self.get_rotation_matrix()
        cylinder_mesh.rotate(rotation_matrix.T, center=(0, 0, 0))
        cylinder_mesh.compute_vertex_normals()
        center = (np.array(self.top_center) + np.array(self.bottom_center)) / 2
        cylinder_mesh.translate(center)
        return cylinder_mesh

    def get_minimum_bounding_box(self) -> 'np.ndarray[np.float32, (8, 3)]':
        # TODO: proceed quickly
        return np.asarray(self.to_o3d_mesh().get_minimal_bounding_box().get_box_points())

    def set_points(self, points: 'np.ndarray[np.float32, (N, 3)]') -> None:
        self.points = points

    def get_reverse(self) -> 'Cylinder':
        new = Cylinder(self.bottom_center, self.top_center, self.radius)
        new.set_points(self.points)
        return new

    def to_segment(self) -> 'Segment':
        return Segment(self.top_center, self.bottom_center)

    def __str__(self) -> str:
        return f"Cylinder({self.top_center}, {self.bottom_center}, {self.radius})"

    @classmethod
    def can_merge_together(cls, cylinder_x: 'Cylinder', cylinder_y: 'Cylinder', angle_threshold: float,
                           distance_threshold: float, radius_threshold_rate: float) -> bool:

        segment_x, segment_y = cylinder_x.to_segment(), cylinder_y.to_segment()

        if Segment.get_angle_from_two_segments(segment_x, segment_y) > angle_threshold:
            return False

        nearest_segment = Segment.get_closest_points_from_two_segments(segment_x, segment_y, False)

        if Segment.get_angle_from_another_segment(nearest_segment, segment_x) > angle_threshold or \
                Segment.get_angle_from_another_segment(nearest_segment, segment_y) > angle_threshold:
            return False

        if abs(cylinder_x.radius - cylinder_y.radius) > radius_threshold_rate * max(cylinder_x.radius,
                                                                                    cylinder_y.radius):
            return False

        if nearest_segment.get_length() > distance_threshold:
            return False

        return True

    @classmethod
    def save_cylinders(cls, cylinders: 'list[Cylinder]', save_prefix: str) -> None:
        cylinder_list, points_list = [], []
        for i, cylinder in enumerate(cylinders):
            _data = cylinder.split_get()
            _data = [*_data, i]
            cylinder_list.append(_data)
            _points = np.hstack((cylinder.points, np.ones((cylinder.points.shape[0], 1)) * i))
            points_list.append(_points)
        cylinder_list = np.vstack(cylinder_list)
        points_list = np.vstack(points_list)
        np.save(f'{save_prefix}_cylinders.npy', cylinder_list)
        np.save(f'{save_prefix}_points.npy', points_list)

    @classmethod
    def load_cylinders(cls, save_prefix: str) -> 'list[Cylinder]':
        cylinder_list = np.load(f'{save_prefix}_cylinders.npy')

        import os, sys
        if os.path.exists(f'{save_prefix}_points.npy'):
            points_list = np.load(f'{save_prefix}_points.npy')
        else:
            print(f'Warning: {save_prefix}_points.npy not found, points loading skipped.', file=sys.stderr)

        cylinders = []
        for i in range(cylinder_list.shape[0]):
            cylinder = Cylinder(cylinder_list[i, :3], cylinder_list[i, 3:6], cylinder_list[i, 6])
            if points_list is not None:
                cylinder.set_points(points_list[points_list[:, -1] == i, :-1])
            cylinders.append(cylinder)
        return cylinders

    @classmethod
    def load_cylinders_from_json(cls, json_file: str) -> 'list[Cylinder]':
        import json
        with open(json_file, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'cylinders' in data:
            data = data['cylinders']
        assert isinstance(data, list)
        cylinders = []
        for item in data:
            cylinder = cls(np.array(item['top_center']), np.array(item['bottom_center']), item['radius'],item['id'],item['anchor_torus_idx'])
            cylinders.append(cylinder)
        return cylinders

    @classmethod
    def get_cylinder_from_segment(cls, segment: 'Segment', radius: float) -> 'Cylinder':
        return Cylinder(segment.p1, segment.p2, radius)

    @classmethod
    def stupid_merge(cls, cylinder_x: 'Cylinder', cylinder_y: 'Cylinder', distance_gap: float = None) -> 'Cylinder':
        """
        将两个圆柱合并成一个圆柱。
        当两个圆柱长度比值没有超过 distance_gap 时，取两个圆柱四端匹配中距离最长的一对点作为新圆柱的两端。
        否则，按照短圆柱在长圆柱上的投影，再取四端匹配中距离最长的一对点作为新圆柱的两端。
        """

        if cylinder_x.get_height() < cylinder_y.get_height():
            cylinder_x, cylinder_y = cylinder_y, cylinder_x

        if distance_gap is not None and distance_gap < 1:
            distance_gap = 1 / distance_gap

        segment_x, segment_y = cylinder_x.to_segment(), cylinder_y.to_segment()

        if distance_gap is not None and cylinder_x.get_height() / cylinder_y.get_height() > distance_gap:
            segment_y = segment_x.get_segment_projection_on_segment(segment_y)

        segment = Segment.get_furthest_points_from_two_segments(segment_x, segment_y, allow_both_side=False)

        cylinder = cls.get_cylinder_from_segment(segment, (cylinder_x.radius + cylinder_y.radius) / 2)
        cylinder.set_points(np.vstack((cylinder_x.points, cylinder_y.points)))

        return cylinder