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


def get_gt_closures(
    dataset, gt_poses: List[np.ndarray], max_range: float, overlap_threshold: float = 0.5
):
    base_dir = dataset.sequence_dir if hasattr(dataset, "sequence_dir") else ""
    os.makedirs(os.path.join(base_dir, "loop_closure"), exist_ok=True)
    file_path_closures = os.path.join(base_dir, "loop_closure", "gt_closures.txt")
    file_path_overlaps = os.path.join(base_dir, "loop_closure", "gt_overlaps.txt")
    if os.path.exists(file_path_closures) and os.path.exists(file_path_overlaps):
        closures = np.loadtxt(file_path_closures, dtype=int)
        overlaps = np.loadtxt(file_path_overlaps)
        assert len(closures) == len(overlaps)
        print(f"[INFO] Found closure ground truth at {file_path_closures}")

    else:
        print("[INFO] Computing Ground Truth Closures, might take some time!")
        min_overlap = 0.1
        sampling_distance = 2.0
        n_skip_segments = int(2 * max_range / sampling_distance)
        segment_indices = get_segment_indices(gt_poses, sampling_distance)
        global_points_segments = [
            get_global_points(dataset, segment, gt_poses[segment]) for segment in segment_indices
        ]
        closures = np.empty((0, 2), dtype=int)
        overlaps = np.empty((0,), dtype=float)
        tbar = tqdm(
            zip(segment_indices, global_points_segments),
            total=len(segment_indices),
        )
        for i, (query_segment, query_points) in enumerate(tbar):
            query_pose = gt_poses[query_segment[0]]
            for _, (ref_segment, ref_points) in enumerate(
                zip(
                    segment_indices[i + n_skip_segments :],
                    global_points_segments[i + n_skip_segments :],
                ),
                start=n_skip_segments + 1,
            ):
                ref_pose = gt_poses[ref_segment[0]]
                dist = np.linalg.norm(query_pose[:3, -1] - ref_pose[:3, -1])
                if dist < max_range:
                    overlap = compute_overlap(query_points, ref_points, voxel_size=0.5)
                    if overlap > min_overlap:
                        segment_closures = np.dstack(
                            np.meshgrid(query_segment, ref_segment)
                        ).reshape(-1, 2)
                        closures = np.vstack((closures, segment_closures), dtype=int)
                        overlaps = np.hstack(
                            (overlaps, np.full(segment_closures.shape[0], overlap))
                        )
        np.savetxt(file_path_closures, closures)
        np.savetxt(file_path_overlaps, overlaps)

    closures = closures[np.where(overlaps > overlap_threshold)[0]]
    return closures
