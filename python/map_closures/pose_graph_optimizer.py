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


class PoseGraphOptimizer:
    def __init__(self, max_iterations: int):
        self.pgo = map_closures_pybind._PoseGraphOptimizer(max_iterations)

    def add_variable(self, id_: int, pose: np.ndarray):
        self.pgo._add_variable(id_, pose)

    def fix_variable(self, id_: int):
        self.pgo._fix_variable(id_)

    def add_factor(
        self, id_source, id_target, relative_pose, information_matrix, robust_kernel=False
    ):
        self.pgo._add_factor(id_source, id_target, relative_pose, information_matrix, robust_kernel)

    def optimize(self):
        self.pgo._optimize()

    def estimates(self):
        return self.pgo._estimates()
