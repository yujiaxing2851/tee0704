import math
from typing import Tuple

import numpy as np


class Segment:
    def __init__(self, p1: 'np.ndarray[(3, 1), np.float32]', p2: 'np.ndarray[(3, 1), np.float32]'):
        self.p1 = p1
        self.p2 = p2

    def get(self) -> 'np.ndarray[(3, 1), np.float32], np.ndarray[(3, 1), np.float32]':
        return self.p1, self.p2

    def split_get(self) -> 'Tuple[float, float, float, float, float, float]':
        return self.p1[0], self.p1[1], self.p1[2], self.p2[0], self.p2[1], self.p2[2]

    def numpy_get(self) -> 'np.ndarray[(6, 1), np.float32]':
        return np.array([self.p1[0], self.p1[1], self.p1[2], self.p2[0], self.p2[1], self.p2[2]])

    def __str__(self) -> str:
        return f"Segment({self.p1}, {self.p2})"

    def get_direction(self) -> 'np.ndarray[(3, 1), np.float32]':
        delta = np.array(self.p2) - np.array(self.p1)
        return delta / np.linalg.norm(delta)

    def get_length(self) -> float:
        return np.linalg.norm(np.array(self.p2) - np.array(self.p1))

    def get_reverse(self):
        return Segment(self.p2, self.p1)

    def get_point_projection_on_segment(self, point: 'np.ndarray[(3, 1), np.float32]') \
            -> 'np.ndarray[(3, 1), np.float32]':
        p1x, p1y, p1z, p2x, p2y, p2z = self.split_get()
        px, py, pz = point
        Delta = np.array([(p2x - p1x), (p2y - p1y), (p2z - p1z)])
        Theta = np.array([(px - p1x), (py - p1y), (pz - p1z)])
        a = np.dot(Theta, Delta) / np.dot(Delta, Delta)
        a = np.clip(a, 0, 1)
        return np.array([p1x, p1y, p1z]) + a * Delta

    def get_segment_projection_on_segment(self, another_segment: 'Segment') -> 'Segment':
        a = self.get_point_projection_on_segment(another_segment.p1)
        b = self.get_point_projection_on_segment(another_segment.p2)
        return Segment(a, b)

    def get_distance_from_another_segment(self, another_segment: 'Segment') -> float:
        p1x, p1y, p1z, p2x, p2y, p2z = self.split_get()
        q1x, q1y, q1z, q2x, q2y, q2z = another_segment.split_get()

        theta = np.array([(q2x - q1x), (q2y - q1y), (q2z - q1z)]) * (-1)
        delta = np.array([(p2x - p1x), (p2y - p1y), (p2z - p1z)]) * (-1)
        gamma = np.array([(p2x - q2x), (p2y - q2y), (p2z - q2z)])

        numerator_a = ((np.dot(gamma, theta) * np.dot(theta, delta)) / np.dot(theta, theta)) - np.dot(gamma, delta)
        numerator_b = np.dot(gamma, theta) - (np.dot(gamma, delta) * np.dot(theta, delta) / np.dot(delta, delta))

        denominator_a = np.dot(delta, delta) - (np.dot(theta, delta) ** 2 / np.dot(theta, theta))
        denominator_b = np.dot(theta, theta) - (np.dot(theta, delta) ** 2 / np.dot(delta, delta))

        # 考虑严格平行问题 非GT不考虑 如存在除以零bug考虑增加平行线段计算
        if denominator_a == 0:
            return np.linalg.norm(gamma)

        a, b = numerator_a / denominator_a, numerator_b / denominator_b
        a, b = np.clip([a, b], 0, 1)
        min_distance = np.dot(gamma + a * delta - b * theta, gamma + a * delta - b * theta) ** 0.5
        return min_distance

    @classmethod
    def get_distance_from_two_segments(cls, segment1: 'Segment', segment2: 'Segment') -> float:
        return segment1.get_distance_from_another_segment(segment2)

    def get_angle_from_another_segment(self, another_segment: 'Segment') -> float:
        p1x, p1y, p1z, p2x, p2y, p2z = self.split_get()
        q1x, q1y, q1z, q2x, q2y, q2z = another_segment.split_get()

        theta = np.array([(q2x - q1x), (q2y - q1y), (q2z - q1z)])
        delta = np.array([(p2x - p1x), (p2y - p1y), (p2z - p1z)])

        theta_gt = theta / ((np.dot(theta, theta)) ** 0.5)
        delta_gt = delta / ((np.dot(delta, delta)) ** 0.5)
        angle = math.acos(np.dot(theta_gt, delta_gt))
        if angle > math.pi / 2:
            angle = math.pi - angle
        return angle

    @classmethod
    def get_angle_from_two_segments(cls, segment1: 'Segment', segment2: 'Segment') -> float:
        return segment1.get_angle_from_another_segment(segment2)

    @classmethod
    def get_closest_points_from_two_segments(cls, segment1: 'Segment', segment2: 'Segment',
                                             allow_both_side: bool = False) -> 'Segment':
        """
        获取两条线段四个顶点中最近的两个点（当 allow_both_side 为 False 时限制不能选取一条线段两端）
        """
        p1, p2 = segment1.get()
        q1, q2 = segment2.get()

        matches = [
            [p1, q1],
            [p1, q2],
            [p2, q1],
            [p2, q2]
        ]

        if allow_both_side:
            matches += [[p1, p2], [q1, q2]]

        distances = [np.linalg.norm(np.array(p) - np.array(q)) for p, q in matches]
        p, q = matches[np.argsort(distances)[0]]

        return Segment(p, q)

    @classmethod
    def get_furthest_points_from_two_segments(cls, segment1: 'Segment', segment2: 'Segment',
                                              allow_both_side: bool = False) -> 'Segment':
        """
        获取两条线段四个顶点中最远的两个点（当 allow_both_side 为 False 时限制不能选取一条线段两端）
        """
        p1, p2 = segment1.get()
        q1, q2 = segment2.get()

        matches = [
            [p1, q1],
            [p1, q2],
            [p2, q1],
            [p2, q2]
        ]

        if allow_both_side:
            matches += [[p1, p2], [q1, q2]]

        distances = [np.linalg.norm(np.array(p) - np.array(q)) for p, q in matches]
        p, q = matches[np.argsort(distances)[-1]]

        return Segment(p, q)

    @classmethod
    def dis_from_two_points(cls, point_x: 'Tuple[float, float, float]', point_y: 'Tuple[float, float, float]') -> float:
        return np.linalg.norm(np.array(point_x) - np.array(point_y))


if __name__ == '__main__':
    segment1 = Segment(np.array([0, 0, 0]), np.array([1, 0, 0]))
    segment2 = Segment(np.array([1, 1, 1]), np.array([1, 1, 2]))
    print(Segment.get_distance_from_two_segments(segment1, segment2))
    print(Segment.get_angle_from_two_segments(segment1, segment2))
    print(Segment.get_closest_points_from_two_segments(segment1, segment2, False))
    print(Segment.get_closest_points_from_two_segments(segment1, segment2, True))
