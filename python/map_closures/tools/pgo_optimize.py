# MIT License
#
# Copyright (c) 2024 Saurabh Gupta, Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch,
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
import os
import sys
import importlib
from abc import ABC
from typing import List

import numpy as np
from tqdm.auto import trange

from pgo.pose_graph_optimizer import PoseGraphOptimizer
from map_closures.tools.evaluation import LocalMap


class StubOptimizer(ABC):
    def __init__(self, gt_poses):
        pass

    def optimize(self, closures, local_maps, odom_poses):
        pass

    def _log_to_file(self, output_dir):
        pass

    def _plot_trajectories(self):
        pass


class Optimizer(StubOptimizer):
    def __init__(self, gt_poses: np.ndarray):
        try:
            self.o3d = importlib.import_module("open3d")
        except ModuleNotFoundError:
            print(
                '[ERROR] This dataloader requires open3d but is not installed on your system run "pip install open3d"'
            )
            sys.exit(1)
        self.gt_poses = (
            np.einsum("...ij,...jk->...ik", np.linalg.inv(gt_poses[0]), gt_poses)
            if gt_poses is not None
            else gt_poses
        )

        self.voxel_size = 1.0
        self.optimizer = PoseGraphOptimizer()

    def _gicp(self, source: np.ndarray, target: np.ndarray, initial_guess: np.ndarray):
        distance_threshold = self.voxel_size * 0.4
        result = self.o3d.pipelines.registration.registration_generalized_icp(
            self.o3d.geometry.PointCloud(self.o3d.utility.Vector3dVector(source)),
            self.o3d.geometry.PointCloud(self.o3d.utility.Vector3dVector(target)),
            distance_threshold,
            initial_guess,
            self.o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        )
        return result

    def optimize(
        self, closures: List[np.ndarray], local_maps: List[LocalMap], odom_poses: np.ndarray
    ):
        self.closures = closures
        self.local_maps = local_maps
        self.odom_poses = odom_poses
        for idx, pose in enumerate(self.odom_poses):
            self.optimizer.add_variable(idx, pose)
        self.optimizer.fix_variable(0)
        omega_poses = np.eye(6)
        for idx in range(len(self.odom_poses) - 1):
            Ti = self.odom_poses[idx]
            Tj = self.odom_poses[idx + 1]
            self.optimizer.add_factor(idx, idx + 1, np.linalg.inv(Tj) @ Ti, omega_poses)

        omega_closure = 1e3 * np.eye(6)

        print("\nRunning g2o optimization on detected closures\n")
        for closure_idx in trange(
            0, len(self.closures), ncols=8, unit=" closures", dynamic_ncols=True
        ):
            closure = self.closures[closure_idx]
            (
                local_map_source,
                local_map_target,
                first_scan_in_source,
                first_scan_in_target,
            ) = closure[:4].astype(int)
            initial_guess = closure[4:].reshape(4, 4)

            source = self.local_maps[local_map_source].pointcloud
            target = self.local_maps[local_map_target].pointcloud
            estimate = self._gicp(source, target, initial_guess)
            if estimate.fitness > 0.5:
                self.optimizer.add_factor(
                    first_scan_in_source,
                    first_scan_in_target,
                    estimate.transformation,
                    omega_closure,
                )

        self.optimizer.optimize()
        optimized_poses = dict(self.optimizer.estimates())
        self.g2o_poses = np.ones((len(optimized_poses), 4, 4))
        for idx, pose in optimized_poses.items():
            self.g2o_poses[idx] = pose
        self.g2o_poses = np.einsum(
            "...ij,...jk->...ik", np.linalg.inv(self.g2o_poses[0]), self.g2o_poses
        )

    def _log_to_file(self, output_dir: str):
        self.optimizer.write_graph(os.path.join(output_dir, "out.g2o"))
        np.savetxt(
            os.path.join(output_dir, "g2o_poses_kitti.txt"), self.g2o_poses[:, :3].reshape(-1, 12)
        )

    def _plot_trajectories(self):
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(10, 6), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        plt.plot(
            self.odom_poses[:, 0, -1],
            self.odom_poses[:, 1, -1],
            self.odom_poses[:, 2, -1],
            linewidth=2,
            color="tab:red",
        )
        plt.plot(
            self.g2o_poses[:, 0, -1],
            self.g2o_poses[:, 1, -1],
            self.g2o_poses[:, 2, -1],
            linewidth=2,
            color="tab:blue",
        )
        if self.gt_poses is not None:
            plt.plot(
                self.gt_poses[:, 0, -1],
                self.gt_poses[:, 1, -1],
                self.gt_poses[:, 2, -1],
                linewidth=2,
                color="tab:green",
            )
            ax.legend(["odometry", "g2o-optimized", "groundtruth"], fontsize=20)
        else:
            ax.legend(["odometry", "g2o-optimized"], fontsize=20)

        plt.axis("auto")
        plt.show()
