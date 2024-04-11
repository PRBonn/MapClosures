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
import os
from typing import List

import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.mapping import VoxelHashMap
from tqdm import tqdm


def get_segment_indices(poses, sampling_distance):
    segments = []
    current_segment = []
    last_pose = poses[0]
    traveled_distance = 0.0
    for index, pose in enumerate(poses):
        traveled_distance += np.linalg.norm(last_pose[:3, -1] - pose[:3, -1])
        if traveled_distance > sampling_distance:
            segments.append(current_segment)

            # Start new segment
            current_segment = [index]
            traveled_distance = 0.0
        else:
            current_segment.append(index)
        last_pose = pose

    segments.append(current_segment)
    return segments


def get_global_points(dataset, indices, poses):
    global_points = []
    for index, pose in zip(indices, poses):
        try:
            points, _ = dataset[index]
        except ValueError:
            points = dataset[index]
        R = pose[:3, :3]
        t = pose[:3, -1]
        global_points.append(points @ R.T + t)
    return np.concatenate(global_points, axis=0)


def compute_overlap(query_points: np.ndarray, ref_points: np.ndarray, voxel_size: float):
    # Compute Szymkiewicz–Simpson coefficient
    # See https://en.wikipedia.org/wiki/Overlap_coefficient
    query_map = VoxelHashMap(voxel_size, 1, 1)
    query_map.add_points(query_points)
    n_query_voxels = len(query_map.point_cloud())

    ref_map = VoxelHashMap(voxel_size, 1, 1)
    ref_map.add_points(ref_points)
    n_ref_voxels = len(ref_map.point_cloud())

    ref_map.add_points(query_points)
    union = len(ref_map.point_cloud())
    intersection = n_query_voxels + n_ref_voxels - union
    overlap = intersection / min(n_query_voxels, n_ref_voxels)
    return overlap


def get_gt_closures(dataset, gt_poses: List[np.ndarray], config: KISSConfig):
    base_dir = dataset.sequence_dir if hasattr(dataset, "sequence_dir") else ""
    file_path_closures = os.path.join(base_dir, "loop_closure/gt_closures.txt")
    file_path_overlaps = os.path.join(base_dir, "loop_closure/gt_overlaps.txt")
    if os.path.exists(file_path_closures) and os.path.exists(file_path_overlaps):
        closures = np.loadtxt(file_path_closures, dtype=int)
        overlaps = np.loadtxt(file_path_overlaps)
        assert len(closures) == len(overlaps)
        print(f"[INFO] Found closure ground truth at {file_path_closures}")

    else:
        min_overlap = 0.1
        sampling_distance = 2.0
        segment_indices = get_segment_indices(gt_poses, sampling_distance)
        closures = []
        overlaps = []
        tbar = tqdm(segment_indices, desc="Computing Ground Truth Closures, might take some time!")
        for i, query_segment in enumerate(tbar):
            query_pose = gt_poses[query_segment[0]]
            query_points = get_global_points(dataset, query_segment, gt_poses[query_segment])
            n_skip_segments = int(2 * config.data.max_range / sampling_distance)
            for _, ref_segment in enumerate(
                segment_indices[i + n_skip_segments :], start=n_skip_segments + 1
            ):
                ref_pose = gt_poses[ref_segment[0]]
                dist = np.linalg.norm(query_pose[:3, -1] - ref_pose[:3, -1])
                if dist < config.data.max_range:
                    ref_points = get_global_points(dataset, ref_segment, gt_poses[ref_segment])
                    overlap = compute_overlap(query_points, ref_points, voxel_size=0.5)
                    if overlap > min_overlap:
                        for qi in query_segment:
                            for ri in ref_segment:
                                closures.append({qi, ri})
                                overlaps.append(overlap)
        np.savetxt(file_path_closures, np.array(closures, dtype=int))
        np.savetxt(file_path_overlaps, np.array(overlaps))
    return closures, overlaps
