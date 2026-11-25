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
import datetime
import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP
from tqdm.auto import trange

from map_closures.config import load_config, write_config
from map_closures.map_closures_multisession import MapClosuresMultisession
from map_closures.tools.evaluation import EvaluationPipeline, StubEvaluation
from map_closures.tools.pgo_optimize import MultiSessionOptimizer, StubOptimizer
from map_closures.tools.utils import Data
from map_closures.voxel_map import VoxelMap


class MultiSessionMapClosurePipeline:
    def __init__(
        self,
        dataset_query,
        ref_data_path: Path,
        config_path: Path,
        results_dir: Path,
        eval: Optional[bool] = False,
        vis: Optional[bool] = False,
        opt: Optional[bool] = False,
    ):
        self._query_dataset = dataset_query
        self._query_dataset_name = (
            self._query_dataset.sequence_id
            if hasattr(self._query_dataset, "sequence_id")
            else os.path.basename(self._query_dataset.data_dir)
        )
        self._ref_data_path = str(ref_data_path)
        self._n_scans = len(self._query_dataset)
        self._results_dir = results_dir
        self._eval = eval
        self._vis = vis
        self._opt = opt

        self.closure_config = load_config(config_path)
        self.map_closures_multisession = MapClosuresMultisession(self.closure_config)

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
        self._multisession_closures: List[np.ndarray] = []

        # Load reference session data and inject into map_closures
        self._load_reference_data()

        # Evaluation
        self.results = StubEvaluation()
        if self._eval:
            gt_closures_path = os.path.join(
                self._ref_data_path,
                "loop_closure",
                f"{self._query_dataset_name}_local_map_gt_closures.txt",
            )
            if os.path.exists(gt_closures_path):
                gt_closures = np.loadtxt(gt_closures_path, dtype=int)
                self.results = EvaluationPipeline(gt_closures, self._query_dataset_name)

    def run(self):
        self._run_pipeline()
        self.results.compute_metrics()
        self.optimized_poses = self._run_pose_graph_optimization()
        self._save_metadata()
        self._plot_trajectories()
        return self.results

    def _load_reference_data(self):
        self.map_closures_multisession.load_hbst_database(
            os.path.join(self._ref_data_path, "hbst_database.bin")
        )

        with open(os.path.join(self._ref_data_path, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)

        self.ref_scan_index_range = metadata["local_maps_scan_index_range"]

        ground_alignment_transforms = np.asarray(metadata["ground_alignment_transforms"])
        for i, T in enumerate(ground_alignment_transforms):
            self.map_closures_multisession.set_reference_ground_alignment(i, T)

        self.ref_poses = np.asarray(metadata["optimized_poses"])
        self.ref_local_maps = metadata["local_maps"]

    def _run_pipeline(self):
        map_idx = 0
        local_map_start_scan_idx = 0
        current_map_pose = np.eye(4)

        for scan_idx in trange(
            0,
            self._n_scans,
            ncols=80,
            unit=" frames",
            dynamic_ncols=True,
            desc="Multi-Session Loop Closure Detection",
        ):
            raw_frame, timestamps = self._query_dataset[scan_idx]
            deskewed_frame, _ = self.odometry.register_frame(raw_frame, timestamps)
            self.odom_poses[scan_idx] = self.odometry.last_pose
            current_frame_pose = self.odometry.last_pose

            frame_to_map_pose = np.linalg.inv(current_map_pose) @ current_frame_pose
            self.voxel_local_map.integrate_frame(deskewed_frame, frame_to_map_pose)

            if np.linalg.norm(frame_to_map_pose[:3, -1]) > self._map_range or (
                scan_idx == self._n_scans - 1
            ):
                local_map_pointcloud = self.voxel_local_map.point_cloud()
                voxel_means, voxel_normals = self.voxel_local_map.per_voxel_mean_and_normal()

                closures = self.map_closures_multisession.get_closures(
                    map_idx, local_map_pointcloud, voxel_means, voxel_normals
                )

                self.data.append_localmap(
                    local_map_pointcloud,
                    None,
                    None,
                    [local_map_start_scan_idx, scan_idx + 1],
                )

                for closure in closures:
                    if closure.number_of_inliers > self.closure_config.inliers_threshold:
                        self.data.append_closure(closure)
                        self.results.append(
                            closure.source_id, closure.target_id, closure.number_of_inliers
                        )

                self.voxel_local_map.remove_far_away_points(frame_to_map_pose[:3, -1])
                pts_to_keep = self.voxel_local_map.point_cloud()
                self.voxel_local_map.clear()
                self.voxel_local_map.integrate_frame(
                    pts_to_keep, np.linalg.inv(current_frame_pose) @ current_map_pose
                )
                current_map_pose = np.copy(current_frame_pose)
                local_map_start_scan_idx = scan_idx
                map_idx += 1

    def _run_pose_graph_optimization(self):
        pgo_optimizer = StubOptimizer()
        optimized_poses = None
        if self._opt and self.data.closures:
            pgo_optimizer = MultiSessionOptimizer()
            pgo_optimizer.add_reference_data(
                self.ref_poses, self.ref_local_maps, self.ref_scan_index_range
            )
            optimized_poses = pgo_optimizer.optimize(
                self.data.closures,
                self.data.local_maps,
                self.data.local_maps_scan_index_range,
                self.odom_poses,
            )

        return optimized_poses

    def _save_metadata(self):
        self._results_dir = self._create_results_dir()
        np.savetxt(
            os.path.join(self._results_dir, "multisession_closures.txt"),
            np.asarray(self._multisession_closures),
        )
        np.savetxt(
            os.path.join(self._results_dir, "kiss_poses_kitti.txt"),
            self.odom_poses[:, :3].reshape(-1, 12),
        )

        if self.optimized_poses is not None:
            np.savetxt(
                os.path.join(self._results_dir, "optimized_poses_kitti.txt"),
                self.optimized_poses[:, :3].reshape(-1, 12),
            )

        self.results.log_to_file(self._results_dir)

    def _plot_trajectories(self):
        from matplotlib import pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(
            self.ref_poses[:, 0, 3],
            self.ref_poses[:, 1, 3],
            color="black",
            label="Reference session optimized poses",
        )
        ax.plot(
            self.odom_poses[:, 0, 3],
            self.odom_poses[:, 1, 3],
            color="blue",
            label="Query session odometry poses",
        )
        if self.optimized_poses is not None:
            ax.plot(
                self.optimized_poses[:, 0, 3],
                self.optimized_poses[:, 1, 3],
                color="red",
                label="Query session optimized poses",
            )

        for i, closure in enumerate(self.data.closures):
            source_idx = int(self.ref_scan_index_range[int(closure[0])][0])
            target_idx = int(self.data.local_maps_scan_index_range[int(closure[1])][0])
            ax.plot(
                [self.ref_poses[source_idx, 0, 3], self.odom_poses[target_idx, 0, 3]],
                [self.ref_poses[source_idx, 1, 3], self.odom_poses[target_idx, 1, 3]],
                color="green",
                linestyle="--",
                linewidth=0.5,
                alpha=0.5,
                label="Cross session loop closure" if i == 0 else "_nolegend_",
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        plt.savefig(os.path.join(self._results_dir, "trajectories.pdf"))

    def _create_results_dir(self) -> Path:
        def get_timestamp() -> str:
            return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        results_dir = os.path.join(
            self._results_dir,
            f"{self._query_dataset_name}_multisession_results",
            get_timestamp(),
        )
        latest_dir = os.path.join(
            self._results_dir, f"{self._query_dataset_name}_multisession_results", "latest"
        )

        os.makedirs(results_dir, exist_ok=True)
        os.unlink(latest_dir) if os.path.exists(latest_dir) or os.path.islink(latest_dir) else None
        os.symlink(results_dir, latest_dir)

        return results_dir
