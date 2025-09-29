# MIT License
#
# Copyright (c) 2025 Tiziano Guadagnino, Benedikt Mersch, Saurabh Gupta, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np

from map_closures.pybind import map_closures_pybind
from map_closures.pybind.map_closures_pybind import _Vector3dVector as Vector3dVector


class VoxelMap:
    def __init__(self, voxel_size: float, max_distance: float):
        self.map = map_closures_pybind._VoxelMap(voxel_size, max_distance)

    def integrate_frame(self, points: np.ndarray, pose: np.ndarray):
        vector3dvector = Vector3dVector(points.astype(np.float64))
        self.map._integrate_frame(vector3dvector, pose)

    def add_points(self, points: np.ndarray):
        vector3dvector = Vector3dVector(points.astype(np.float64))
        self.map._add_points(vector3dvector)

    def point_cloud(self):
        return np.asarray(self.map._point_cloud()).astype(np.float64)

    def clear(self):
        self.map._clear()

    def num_voxels(self):
        return self.map._num_voxels()

    def remove_far_away_points(self, origin):
        self.map._remove_far_away_points(origin)

    def per_voxel_mean_and_normal(self):
        means, normals = self.map._per_voxel_mean_and_normal()
        return np.asarray(means, np.float64), np.asarray(normals, np.float64)
