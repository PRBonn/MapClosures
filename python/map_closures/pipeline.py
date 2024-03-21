# MIT License
#
# Copyright (c) 2023 Saurabh Gupta, Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch,
# Cyrill Stachniss.
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
import datetime
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP
from kiss_icp.mapping import get_voxel_hash_map
from kiss_icp.voxelization import voxel_down_sample
from tqdm.auto import trange

from map_closures.config import MapClosuresConfig, load_config, write_config
from map_closures.map_closures import MapClosures
from map_closures.tools.local_maps import LocalMap


def transform_points(pcd, T):
    R = T[:3, :3]
    t = T[:3, -1]
    return pcd @ R.T + t


class MapClosurePipeline:
    def __init__(
        self,
        dataset,
        config_path: Path,
        results_dir: Path,
        eval: Optional[bool] = False,
    ):
        self._dataset = dataset
        self._dataset_name = (
            self._dataset.sequence_id
            if hasattr(self._dataset, "sequence_id")
            else os.path.basename(self._dataset.data_dir)
        )
        self._n_scans = len(self._dataset)
        self._results_dir = results_dir
        self._eval = eval

        if config_path is not None:
            self.config_name = os.path.basename(config_path)
            self.closure_config = load_config(config_path)
        else:
            self.config_name = "base_config.yaml"
            self.closure_config = MapClosuresConfig()

        self.kiss_config = KISSConfig()
        self.kiss_config.mapping.voxel_size = 1.0
        self.odometry = KissICP(self.kiss_config)
        self.voxel_local_map = get_voxel_hash_map(self.kiss_config)

        self._map_range = self.closure_config.local_map_factor * self.kiss_config.data.max_range
        self.map_closures = MapClosures(self.closure_config)

        self.local_maps: List[LocalMap] = []

        self.closures = []

        if hasattr(self._dataset, "gt_poses") and self._eval:
            from map_closures.tools.evaluation import EvaluationPipeline
            from map_closures.tools.gt_closures import get_gt_closures

            self.gt_closures, self.gt_closures_overlap = get_gt_closures(
                self._dataset, self._dataset.gt_poses, self.kiss_config
            )
            self.closure_overlap_threshold = 0.5
            self.gt_closures = self.gt_closures[
                np.where(self.gt_closures_overlap > self.closure_overlap_threshold)[0]
            ]

            self.closure_distance_threshold = 6.0
            self.results = EvaluationPipeline(
                self.gt_closures,
                self._dataset_name,
                self.closure_distance_threshold,
            )
        else:
            self._eval = False
            self.results = None
            if not hasattr(self._dataset, "gt_poses"):
                print("Cannot compute ground truth closures, no ground truth poses available\n")

    def run(self):
        self._run_pipeline()
        if self._eval:
            self._run_evaluation()
            self.results.print()
        self._save_config()
        self._log_to_file()
        self._log_to_console()
        self._write_data_to_disk()

        return self.results

    def _run_pipeline(self):
        map_idx = 0
        poses_in_local_map = []
        scan_indices_in_local_map = []

        current_map_pose = np.eye(4)
        for scan_idx in trange(0, self._n_scans, ncols=8, unit=" frames", dynamic_ncols=True):
            try:
                frame, timestamps = self._dataset[scan_idx]
            except ValueError:
                frame = self._dataset[scan_idx]
                timestamps = np.zeros(len(frame))

            self.odometry.register_frame(frame, timestamps)
            current_frame_pose = self.odometry.poses[-1]

            frame_downsample = voxel_down_sample(frame, self.kiss_config.mapping.voxel_size * 0.5)
            frame_to_map_pose = np.linalg.inv(current_map_pose) @ current_frame_pose
            self.voxel_local_map.add_points(transform_points(frame_downsample, frame_to_map_pose))

            if np.linalg.norm(frame_to_map_pose[:3, -1]) > self._map_range or (
                scan_idx == self._n_scans - 1
            ):
                local_map_pointcloud = self.voxel_local_map.point_cloud()
                matched_map_indices, density_map = self.map_closures.match_and_add_local_map(
                    map_idx, local_map_pointcloud, map_idx // 2
                )

                scan_indices_in_local_map.append(scan_idx)
                poses_in_local_map.append(current_frame_pose)

                self.local_maps.append(
                    LocalMap(
                        local_map_pointcloud,
                        density_map,
                        np.copy(scan_indices_in_local_map),
                        np.copy(poses_in_local_map),
                    )
                )

                for ref_idx in matched_map_indices:
                    if map_idx - ref_idx >= 3:
                        relative_pose, inliers_count = self.map_closures.check_for_closure(
                            ref_idx, map_idx
                        )
                        if inliers_count > self.closure_config.inliers_threshold:
                            self.closures.append(
                                np.r_[
                                    ref_idx,
                                    map_idx,
                                    self.local_maps[ref_idx].scan_indices[0],
                                    self.local_maps[map_idx].scan_indices[0],
                                    relative_pose.flatten(),
                                ]
                            )
                            if self._eval:
                                self.results.append(
                                    self.local_maps[ref_idx],
                                    self.local_maps[map_idx],
                                    relative_pose,
                                    self.closure_distance_threshold,
                                    inliers_count,
                                )

                self.voxel_local_map.remove_far_away_points(frame_to_map_pose[:3, -1])
                pts_to_keep = self.voxel_local_map.point_cloud()
                self.voxel_local_map = get_voxel_hash_map(self.kiss_config)
                self.voxel_local_map.add_points(
                    transform_points(
                        pts_to_keep, np.linalg.inv(current_frame_pose) @ current_map_pose
                    )
                )
                current_map_pose = np.copy(current_frame_pose)
                scan_indices_in_local_map.clear()
                poses_in_local_map.clear()
                scan_indices_in_local_map.append(scan_idx)
                poses_in_local_map.append(current_frame_pose)
                map_idx += 1
            else:
                scan_indices_in_local_map.append(scan_idx)
                poses_in_local_map.append(current_frame_pose)

    def _run_evaluation(self):
        self.results.compute_closures_and_metrics()

    def _log_to_file(self):
        if self._eval:
            self.results.log_to_file_pr(
                os.path.join(self._results_dir, "evaluation_metrics.txt"), self.closure_config
            )
            self.results.log_to_file_closures(
                os.path.join(self._results_dir, "scan_level_closures.npy")
            )
        np.savetxt(os.path.join(self._results_dir, "map_closures.txt"), np.asarray(self.closures))
        np.save(
            os.path.join(self._results_dir, "kiss_poses.npy"),
            np.asarray(self.odometry.poses),
        )

    def _log_to_console(self):
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(box=box.HORIZONTALS, title=f"MapClosures detected for {self._dataset_name}")
        table.caption = f"Detected MapClosures"
        table.add_column("# MapClosure", justify="left", style="cyan")
        table.add_column("Ref Map Index", justify="left", style="magenta")
        table.add_column("Query Map Index", justify="left", style="magenta")
        table.add_column("Relative Translation 2D", justify="right", style="magenta")
        table.add_column("Relative Rotation 2D", justify="right", style="magenta")

        for i, closure in enumerate(self.closures):
            table.add_row(
                f"{i+1}",
                f"{closure[0]}",
                f"{closure[1]}",
                f"[{closure[7]:.4f}, {closure[11]:.4f}] m",
                f"{(np.arctan2(closure[8], closure[9]) * 180 / np.pi):.4f} deg",
            )
        console.print(table)

    def _save_config(self):
        self._results_dir = self._create_results_dir()
        write_config(self.closure_config, os.path.join(self._results_dir, self.config_name))

    def _write_data_to_disk(self):
        import open3d as o3d
        from matplotlib import pyplot as plt

        local_map_dir = os.path.join(self._results_dir, "local_maps")
        density_map_dir = os.path.join(self._results_dir, "density_maps")
        os.makedirs(density_map_dir, exist_ok=True)
        os.makedirs(local_map_dir, exist_ok=True)
        for i, local_map in enumerate(self.local_maps):
            plt.imsave(
                os.path.join(density_map_dir, f"{i:06d}.png"),
                255 - local_map.density_map,
                cmap="gray",
            )
            o3d.io.write_point_cloud(
                os.path.join(local_map_dir, f"{i:06d}.ply"),
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(local_map.pointcloud)),
            )

    def _create_results_dir(self) -> Path:
        def get_timestamp() -> str:
            return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        results_dir = os.path.join(
            self._results_dir,
            f"{self._dataset_name}_results",
            get_timestamp(),
        )
        latest_dir = os.path.join(self._results_dir, f"{self._dataset_name}_results", "latest")

        os.makedirs(results_dir, exist_ok=True)
        os.unlink(latest_dir) if os.path.exists(latest_dir) or os.path.islink(latest_dir) else None
        os.symlink(results_dir, latest_dir)

        return results_dir
