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

    def add_new_localmap(self, map_idx: int, local_map: np.ndarray) -> np.ndarray:
        _local_map = map_closures_pybind._VectorEigen3d(local_map)
        density_map_img = self._pipeline._AddNewLocalMap(map_idx, _local_map)
        return np.asarray(density_map_img, np.uint8)

    def get_ref_map_indices(self, top_n: int) -> np.ndarray:
        ref_map_ids = self._pipeline._GetRefMapIndices(top_n)
        return np.asarray(ref_map_ids)

    def check_for_closure(self, ref_idx: int, query_idx: int) -> Tuple:
        T_init, inliers_count = self._pipeline._CheckForClosure(ref_idx, query_idx)
        return np.asarray(T_init), inliers_count
