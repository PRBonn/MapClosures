# MIT License
#
# Copyright (c) 2026 Saurabh Gupta, Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch,
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
from abc import ABC
from typing import List

import g2opy as g2o
import numpy as np
from tqdm.auto import trange


class StubOptimizer(ABC):
    def __init__(self, *kwargs):
        pass

    def optimize(self, *kwargs):
        pass

    def log_to_file(self, *kwargs):
        pass

    def add_reference_data(self, *kwargs):
        pass

    def plot_trajectories(self):
        pass


class SingleSessionOptimizer(StubOptimizer):
    def __init__(self):
        try:
            self.o3d = importlib.import_module("open3d")
        except ModuleNotFoundError:
            print('[ERROR] open3d is not installed. Run "pip install open3d"')
            sys.exit(1)
        self.omega_poses = np.eye(6)
        self.omega_closure = np.eye(6)

        self.voxel_size = 1.0
        self.pose_graph = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        algorithm = g2o.OptimizationAlgorithmDogleg(solver)
        self.pose_graph.set_algorithm(algorithm)
        self.pose_graph.set_verbose(True)

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
        self,
        closures: List[np.ndarray],
        local_maps: List[np.ndarray],
        local_map_scan_index_range: List[tuple],
        odom_poses: np.ndarray,
    ):
        self.closures = closures
        self.local_maps = local_maps
        self.local_map_scan_index_range = local_map_scan_index_range
        self.odom_poses = odom_poses

        for i, pose in enumerate(self.odom_poses):
            v = g2o.VertexSE3()
            v.set_id(i)
            v.set_estimate(g2o.Isometry3d(pose))
            self.pose_graph.add_vertex(v)

        for idx in range(len(self.odom_poses) - 1):
            Ti = self.odom_poses[idx]
            Tj = self.odom_poses[idx + 1]
            edge = g2o.EdgeSE3()
            edge.set_vertex(1, self.pose_graph.vertex(idx))
            edge.set_vertex(0, self.pose_graph.vertex(idx + 1))
            edge.set_measurement(g2o.Isometry3d(np.linalg.inv(Tj) @ Ti))
            edge.set_information(self.omega_poses)
            self.pose_graph.add_edge(edge)

        print("\nRunning pose graph optimization on detected closures\n")
        for closure_idx in trange(
            0, len(self.closures), ncols=80, unit=" closures", dynamic_ncols=True
        ):
            closure = self.closures[closure_idx]
            local_map_source, local_map_target = closure[:2].astype(int)
            initial_guess = closure[2:].reshape(4, 4)

            source = self.local_maps[local_map_source]
            target = self.local_maps[local_map_target]
            estimate = self._gicp(source, target, initial_guess)
            if estimate.fitness > 0.5:
                edge = g2o.EdgeSE3()
                edge.set_vertex(
                    1, self.pose_graph.vertex(self.local_map_scan_index_range[local_map_source][0])
                )
                edge.set_vertex(
                    0, self.pose_graph.vertex(self.local_map_scan_index_range[local_map_target][0])
                )
                edge.set_measurement(g2o.Isometry3d(estimate.transformation))
                edge.set_information(self.omega_closure)
                self.pose_graph.add_edge(edge)

        self.pose_graph.vertex(0).set_fixed(True)
        self.pose_graph.initialize_optimization()
        self.pose_graph.optimize(20)
        optimized_poses = np.tile(np.eye(4), (len(self.odom_poses), 1, 1))
        for i, v in self.pose_graph.vertices().items():
            optimized_poses[i] = v.estimate().matrix()

        return optimized_poses


class MultiSessionOptimizer(StubOptimizer):
    def __init__(self):
        try:
            self.o3d = importlib.import_module("open3d")
        except ModuleNotFoundError:
            print('[ERROR] open3d is not installed. Run "pip install open3d"')
            sys.exit(1)

        self.omega_poses = np.eye(6)
        self.omega_closure = np.eye(6)
        self.voxel_size = 1.0

        self.pose_graph = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        algorithm = g2o.OptimizationAlgorithmDogleg(solver)
        self.pose_graph.set_algorithm(algorithm)
        self.pose_graph.set_verbose(True)

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

    def add_reference_data(
        self,
        ref_poses: np.ndarray,
        ref_local_maps: List[np.ndarray],
        ref_scan_index_range: List[tuple],
    ):
        self.ref_poses = ref_poses
        self.ref_local_maps = ref_local_maps
        self.ref_scan_index_range = ref_scan_index_range
        for i, pose in enumerate(self.ref_poses):
            v = g2o.VertexSE3()
            v.set_id(i)
            v.set_fixed(True)
            v.set_estimate(g2o.Isometry3d(pose))
            self.pose_graph.add_vertex(v)

        for idx in range(len(self.ref_poses) - 1):
            Ti = self.ref_poses[idx]
            Tj = self.ref_poses[idx + 1]
            edge = g2o.EdgeSE3()
            edge.set_vertex(1, self.pose_graph.vertex(idx))
            edge.set_vertex(0, self.pose_graph.vertex(idx + 1))
            edge.set_measurement(g2o.Isometry3d(np.linalg.inv(Tj) @ Ti))
            edge.set_information(self.omega_poses)
            self.pose_graph.add_edge(edge)

    def optimize(
        self,
        closures: List[np.ndarray],
        local_maps: List[np.ndarray],
        local_map_scan_index_range: List[List[int]],
        odom_poses: np.ndarray,
    ):
        self.offset = self.ref_poses.shape[0]
        self.closures = closures
        self.local_maps = local_maps
        self.local_map_scan_index_range = local_map_scan_index_range
        self.odom_poses = odom_poses

        initial_T = self.closures[0][2:].reshape(4, 4)
        for i, pose in enumerate(self.odom_poses):
            v = g2o.VertexSE3()
            v.set_id(self.offset + i)
            v.set_estimate(g2o.Isometry3d(initial_T * pose))
            self.pose_graph.add_vertex(v)

        for idx in range(len(self.odom_poses) - 1):
            Ti = self.odom_poses[idx]
            Tj = self.odom_poses[idx + 1]
            edge = g2o.EdgeSE3()
            edge.set_vertex(1, self.pose_graph.vertex(self.offset + idx))
            edge.set_vertex(0, self.pose_graph.vertex(self.offset + idx + 1))
            edge.set_measurement(g2o.Isometry3d(np.linalg.inv(Tj) @ Ti))
            edge.set_information(self.omega_poses)
            self.pose_graph.add_edge(edge)

        print("\nRunning pose graph optimization on detected closures\n")
        for closure_idx in trange(
            0, len(self.closures), ncols=80, unit=" closures", dynamic_ncols=True
        ):
            closure = self.closures[closure_idx]
            local_map_source, local_map_target = closure[:2].astype(int)
            initial_guess = closure[2:].reshape(4, 4)

            source = self.ref_local_maps[local_map_source]
            target = self.local_maps[local_map_target]
            estimate = self._gicp(source, target, initial_guess)
            if estimate.fitness > 0.5:
                edge = g2o.EdgeSE3()
                edge.set_vertex(
                    1,
                    self.pose_graph.vertex(self.ref_scan_index_range[local_map_source][0]),
                )
                edge.set_vertex(
                    0,
                    self.pose_graph.vertex(
                        self.local_map_scan_index_range[local_map_target][0] + self.offset
                    ),
                )
                edge.set_measurement(g2o.Isometry3d(estimate.transformation))
                edge.set_information(self.omega_poses)
                self.pose_graph.add_edge(edge)

        self.pose_graph.initialize_optimization()
        self.pose_graph.optimize(20)

        self.g2o_poses = np.tile(np.eye(4), (len(self.odom_poses), 1, 1))
        for i in range(len(self.odom_poses)):
            v = self.pose_graph.vertex(self.offset + i)
            self.g2o_poses[i] = v.estimate().matrix()

    def log_to_file(self, output_dir: str):
        np.savetxt(
            os.path.join(output_dir, "g2o_poses_kitti.txt"),
            self.g2o_poses[:, :3].reshape(-1, 12),
        )
        np.save(os.path.join(output_dir, "g2o_poses.npy"), self.g2o_poses)

    def plot_trajectories(self):
        from matplotlib import pyplot as plt

        fig = plt.figure(figsize=(10, 6), constrained_layout=True)
        ax = fig.add_subplot(111)
        plt.plot(
            self.ref_poses[:, 0, -1],
            self.ref_poses[:, 1, -1],
            linewidth=2,
            color="tab:red",
        )
        plt.plot(
            self.g2o_poses[:, 0, -1],
            self.g2o_poses[:, 1, -1],
            linewidth=2,
            color="tab:blue",
        )
        ax.legend(["Reference-Session", "Query-Session"], fontsize=20)
        plt.axis("auto")
        plt.show()
