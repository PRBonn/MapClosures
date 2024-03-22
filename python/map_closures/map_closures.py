# MIT License
#
# Copyright (c) 2024 Saurabh Gupta, Tiziano Guadagnino, Benedikt Mersch,
# Ignacio Vizzo, Cyrill Stachniss.
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
from typing import Tuple

import numpy as np
from pydantic_settings import BaseSettings

from map_closures.pybind import map_closures_pybind


class MapClosures:
    def __init__(self, config: BaseSettings):
        self._config = config
        self._pipeline = map_closures_pybind._MapClosures(self._config.model_dump())

    def match_and_add_local_map(self, map_idx: int, local_map: np.ndarray, top_k: int):
        _local_map = map_closures_pybind._VectorEigen3d(local_map)
        ref_map_indices, density_map_img = self._pipeline._MatchAndAddLocalMap(
            map_idx, _local_map, top_k
        )
        return np.asarray(ref_map_indices, np.int32), np.asarray(density_map_img, np.uint8)

    def detect_loop_closure_and_add_local_map(self, map_idx, local_map):
        ref_idx, map_idx, relative_tf_2d, inliers_count = self._pipeline(map_idx, local_map)
        closure = {"source_idx": -1, "target_idx": -1, "inliers_count": 0, "pose": np.eye(3)}
        closure["source_idx"] = ref_idx
        closure["target_idx"] = map_idx
        closure["inliers_count"] = inliers_count
        closure["pose"] = relative_tf_2d
        return closure

    def add_local_map_and_compute_closure(self, map_idx: int, local_map: np.ndarray, top_k: int):
        matches, _ = self.match_and_add_local_map(map_idx, local_map, top_k)
        best_closure = {"source_idx": -1, "target_idx": -1, "inliers_count": 0, "pose": np.eye(3)}
        max_inliers_count = 0
        for ref_idx in matches:
            if map_idx - ref_idx < 3:
                continue
            relative_tf_2d, inliers_count = self.check_for_closure(ref_idx, map_idx)
            if inliers_count > max_inliers_count:
                best_closure["source_idx"] = ref_idx
                best_closure["target_idx"] = map_idx
                best_closure["inliers_count"] = inliers_count
                best_closure["pose"] = relative_tf_2d
                max_inliers_count = inliers_count
        return best_closure

    def check_for_closure(self, ref_idx: int, query_idx: int) -> Tuple:
        T_init, inliers_count = self._pipeline._CheckForClosure(ref_idx, query_idx)
        return np.asarray(T_init), inliers_count
