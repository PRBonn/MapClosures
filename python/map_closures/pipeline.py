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
from typing import Optional

import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP
from tqdm.auto import trange

from map_closures.config import load_config, write_config
from map_closures.map_closures import MapClosures
from map_closures.tools.evaluation import EvaluationPipeline, StubEvaluation
from map_closures.tools.utils import Data, pose_inv
from map_closures.visualizer.visualizer import StubVisualizer, Visualizer
from map_closures.voxel_map import VoxelMap


class MapClosurePipeline:
    def __init__(
        self,
        dataset,
        config_path: Path,
        results_dir: Path,
        eval: Optional[bool] = False,
        vis: Optional[bool] = False,
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
        self._vis = vis

        self.closure_config = load_config(config_path)
        self.map_closures = MapClosures(self.closure_config)

        self.kiss_config = KISSConfig()
        self.kiss_config.mapping.voxel_size = 1.0
        self.kiss_config.data.max_range = 100.0
        self.odometry = KissICP(self.kiss_config)

        self.voxel_local_map = VoxelMap(
            voxel_size=self.closure_config.density_map_resolution,
            max_distance=self.kiss_config.data.max_range,
        )
        self._map_range = self.kiss_config.data.max_range

        self.data = Data()
        self.odom_poses = np.zeros((self._n_scans, 4, 4))

        self.results = StubEvaluation()
        if self._eval:
            gt_closures_path = (
                os.path.join(
                    self._dataset.sequence_dir, "loop_closure", "local_map_gt_closures.txt"
                )
                if hasattr(self._dataset, "sequence_dir")
                else None
            )
            if gt_closures_path is not None and os.path.exists(gt_closures_path):
                self.gt_closures = np.loadtxt(gt_closures_path, dtype=int)
                self.results = EvaluationPipeline(self.gt_closures, self._dataset_name)

        self.visualizer = Visualizer() if self._vis else StubVisualizer()

    def run(self):
        self._run_pipeline()
        self.results.compute_metrics()
        self._save_config()
        self._log_to_file()
        self._log_to_console()

        return self.results

    def _run_pipeline(self):
        map_idx = 0
        local_map_start_scan_idx = 0
        current_map_pose = np.eye(4)

        for scan_idx in trange(
            0,
            self._n_scans,
            ncols=8,
            unit=" frames",
            dynamic_ncols=True,
            desc="Processing Scans for Loop Closures",
        ):
            raw_frame, timestamps = self._dataset[scan_idx]
            source, _ = self.odometry.register_frame(raw_frame, timestamps)
            self.odom_poses[scan_idx] = self.odometry.last_pose
            current_frame_pose = self.odometry.last_pose

            frame_to_map_pose = pose_inv(current_map_pose) @ current_frame_pose
            self.voxel_local_map.integrate_frame(source, frame_to_map_pose)
            self.visualizer.update_registration(
                source, self.odometry.local_map.point_cloud(), current_frame_pose
            )

            if np.linalg.norm(frame_to_map_pose[:3, -1]) > self._map_range or (
                scan_idx == self._n_scans - 1
            ):
                local_map_pointcloud = self.voxel_local_map.point_cloud()
                points, normals = self.voxel_local_map.per_voxel_mean_and_normal()
                closures = self.map_closures.get_closures(
                    map_idx, local_map_pointcloud, points, normals
                )

                density_map = self.map_closures.get_density_map_from_id(map_idx)
                self.data.append_localmap(
                    local_map_pointcloud,
                    density_map,
                    self.map_closures.get_ground_alignment_from_id(map_idx),
                    [local_map_start_scan_idx, scan_idx + 1],
                )

                self.visualizer.update_data(local_map_pointcloud, density_map, current_map_pose)

                for closure in closures:
                    if closure.number_of_inliers > self.closure_config.inliers_threshold:
                        self.data.append_closure(closure)
                        self.results.append(
                            closure.source_id, closure.target_id, closure.number_of_inliers
                        )
                        self.visualizer.update_closures(
                            np.asarray(closure.pose), [closure.source_id, closure.target_id]
                        )

                self.voxel_local_map.remove_far_away_points(frame_to_map_pose[:3, -1])
                pts_to_keep = self.voxel_local_map.point_cloud()
                self.voxel_local_map.clear()
                self.voxel_local_map.integrate_frame(
                    pts_to_keep, pose_inv(current_frame_pose) @ current_map_pose
                )
                current_map_pose = np.copy(current_frame_pose)
                frame_to_map_pose = pose_inv(current_map_pose) @ current_frame_pose
                local_map_start_scan_idx = scan_idx
                map_idx += 1

    def _log_to_file(self):
        np.savetxt(
            os.path.join(self._results_dir, "map_closures.txt"), np.asarray(self.data.closures)
        )
        np.savetxt(
            os.path.join(self._results_dir, "kiss_poses_kitti.txt"),
            self.odom_poses[:, :3].reshape(-1, 12),
        )
        self.results.log_to_file(self._results_dir)

    def _log_to_console(self):
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(box=box.HORIZONTALS)
        table.caption = f"Loop Closures Detected Between Local Maps\n"
        table.add_column("# MapClosure", justify="left", style="cyan")
        table.add_column("Ref Map Index", justify="left", style="magenta")
        table.add_column("Query Map Index", justify="left", style="magenta")

        for i, closure in enumerate(self.data.closures):
            table.add_row(f"{i+1}", f"{int(closure[0])}", f"{int(closure[1])}")
        console.print(table)

    def _save_config(self):
        self._results_dir = self._create_results_dir()
        write_config(self.closure_config, os.path.join(self._results_dir, "config"))

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

    def write_data_to_disk(self):
        import open3d as o3d
        from matplotlib import pyplot as plt

        local_map_dir = os.path.join(self._results_dir, "local_maps")
        os.makedirs(local_map_dir, exist_ok=True)
        density_map_dir = os.path.join(self._results_dir, "density_maps")
        os.makedirs(density_map_dir, exist_ok=True)
        for i, [local_map, density_map] in enumerate(
            zip(self.data.local_maps, self.data.density_maps)
        ):
            o3d.io.write_point_cloud(
                os.path.join(local_map_dir, f"{i:06d}.ply"),
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(local_map)),
            )
            plt.imsave(os.path.join(density_map_dir, f"{i:06d}.png"), density_map, cmap="gray")
