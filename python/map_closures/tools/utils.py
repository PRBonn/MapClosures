from dataclasses import dataclass, field
from typing import List

import numpy as np

from map_closures.map_closures import ClosureCandidate


def pose_inv(T):
    T_inv = np.eye(4)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, -1] = -T[:3, :3].T @ T[:3, -1]
    return T_inv


@dataclass
class Data:
    local_maps: list = field(default_factory=list)
    density_maps: list = field(default_factory=list)
    local_maps_scan_index_range: list = field(default_factory=list)
    ground_alignment_transforms: list = field(default_factory=list)
    closures: list = field(default_factory=list)

    def append_localmap(
        self,
        local_map: np.ndarray,
        density_map: np.ndarray,
        ground_alignment_transform: np.ndarray,
        scan_index_range: List[int],
    ):
        self.local_maps.append(local_map)
        self.density_maps.append(density_map)
        self.ground_alignment_transforms.append(ground_alignment_transform)
        self.local_maps_scan_index_range.append(scan_index_range)

    def append_closure(self, closure: ClosureCandidate):
        self.closures.append(np.r_[closure.source_id, closure.target_id, closure.pose.flatten()])
