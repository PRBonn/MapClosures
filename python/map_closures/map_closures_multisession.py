# MIT License
#
# Copyright (c) 2026 Saurabh Gupta
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
from typing import List, overload

import numpy as np
from typing_extensions import TypeAlias

from map_closures.config import MapClosuresConfig
from map_closures.pybind import map_closures_multisession_pybind
from map_closures.pybind.map_closures_multisession_pybind import (
    _Vector3dVector as Vector3dVector,
)

ClosureCandidate: TypeAlias = map_closures_multisession_pybind._ClosureCandidate


class MapClosuresMultisession:
    def __init__(self, config: MapClosuresConfig = MapClosuresConfig()):
        self._config = config
        self._pipeline = map_closures_multisession_pybind._MapClosuresMultisession(
            self._config.model_dump()
        )

    def get_density_map_from_id(self, map_id: int) -> np.ndarray:
        return self._pipeline._getDensityMapFromId(map_id)

    def get_ground_alignment_from_id(self, map_id: int) -> np.ndarray:
        return np.asarray(self._pipeline._getGroundAlignmentFromId(map_id))

    def set_reference_ground_alignment(self, map_id: int, alignment: np.ndarray):
        self._pipeline._setReferenceGroundAlignment(map_id, alignment)

    def load_hbst_database(self, database_path: str):
        self._pipeline._LoadHbstDatabase(database_path)

    def get_closures(
        self,
        query_idx: int,
        local_map: np.ndarray,
        voxel_means: np.ndarray,
        voxel_normals: np.ndarray,
    ) -> List[ClosureCandidate]:
        return self._pipeline._GetClosures(
            query_idx,
            Vector3dVector(local_map),
            Vector3dVector(voxel_means),
            Vector3dVector(voxel_normals),
        )
