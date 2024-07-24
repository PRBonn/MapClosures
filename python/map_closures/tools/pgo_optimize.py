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
import importlib
import os
import sys

import numpy as np
from pgo.pose_graph_optimizer import PoseGraphOptimizer
from tqdm.auto import trange


class PGO_Optimizer:
    def __init__(self, closures, local_maps, odom_poses, gt_poses):
        try:
            self.o3d = importlib.import_module("open3d")
        except ModuleNotFoundError:
            print(
                '[ERROR] This dataloader requires open3d but is not installed on your system run "pip install open3d"'
            )
            sys.exit(1)
        self.closures = closures
        self.local_maps = local_maps
        self.odom_poses = odom_poses
        self.gt_poses = (
            np.einsum("...ij,...jk->...ik", np.linalg.inv(gt_poses[0]), gt_poses)
            if gt_poses is not None
            else gt_poses
        )

        self.voxel_size = 1.0
        self.optimizer = PoseGraphOptimizer()

    def gicp(self, source, target, initial_guess):
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
        self,
    ):
        for idx, pose in enumerate(self.odom_poses):
            self.optimizer.add_variable(idx, pose)
        omega_poses = np.eye(6)
        for idx in range(len(self.odom_poses) - 1):
            Ti = self.odom_poses[idx]
            Tj = self.odom_poses[idx + 1]
            self.optimizer.add_factor(idx, idx + 1, np.linalg.inv(Ti) @ Tj, omega_poses)

        omega_closure = 1e3 * np.eye(6)
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
            estimate = self.gicp(source, target, initial_guess)
            if estimate.fitness > 0.5:
                self.optimizer.add_factor(
                    first_scan_in_target,
                    first_scan_in_source,
                    estimate.transformation,
                    omega_closure,
                )

        self.optimizer.optimize()
        g2o_poses = np.array([pose for pose in dict(self.optimizer.estimates()).values()])
        self.g2o_poses = np.einsum("ij,njk->nik", np.linalg.inv(g2o_poses[0]), g2o_poses)

    def _log_to_file(self, output_dir):
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
        )
        plt.plot(
            self.g2o_poses[:, 0, -1],
            self.g2o_poses[:, 1, -1],
            self.g2o_poses[:, 2, -1],
            linewidth=2,
        )
        if self.gt_poses is not None:
            plt.plot(
                self.gt_poses[:, 0, -1],
                self.gt_poses[:, 1, -1],
                self.gt_poses[:, 2, -1],
                linewidth=2,
            )
            ax.legend(["odometry", "g2o-optimized", "groundtruth"], fontsize=20)
        else:
            ax.legend(["odometry", "g2o-optimized"], fontsize=20)

        plt.axis("auto")
        plt.show()
