# MIT License
#
# Copyright (c) 2024 Saurabh Gupta, Ignacio Vizzo, Tiziano Guadagnino,
# Benedikt Mersch, Cyrill Stachniss.
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
import glob
from pathlib import Path

from pyquaternion import Quaternion
import numpy as np

class HeLiPRDataset:
    def __init__(self, data_dir: Path, sequence: str, *_, **__):
        self.sequence_id = sequence
        self.sequence_dir = os.path.realpath(data_dir)
        self.scans_dir = os.path.join(self.sequence_dir, "LiDAR", self.sequence_id)
        self.scan_files = sorted(glob.glob(self.scans_dir + "/*.bin"))

        self.gt_file = os.path.join(self.sequence_dir, "LiDAR_GT", f"{self.sequence_id}_gt.txt")
        self.gt_poses = self.load_poses(self.gt_file)

        self.dtype = np.dtype(
            [
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("reflectivity", np.uint8),
                ("tag", np.uint8),
                ("line", np.uint8),
                ("offset_time", np.uint32),
            ]
        )

    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, idx):
        return self.read_point_cloud(self.scan_files[idx])

    def read_point_cloud(self, file_path: str):
        points = np.stack(
            [
                [line[0], line[1], line[2]]
                for line in np.fromfile(file_path, dtype=self.dtype).tolist()
            ]
        )
        return points.astype(np.float64)

    def load_poses(self, poses_file):
        poses = np.loadtxt(poses_file, delimiter=" ")
        n = poses.shape[0]

        xyz = poses[:, 1:4]
        rotations = np.array(
            [Quaternion(x=x, y=y, z=z, w=w).rotation_matrix for x, y, z, w in poses[:, 4:]]
        )
        poses = np.eye(4, dtype=np.float64).reshape(1, 4, 4).repeat(self.__len__(), axis=0)
        poses[:, :3, :3] = rotations
        poses[:, :3, 3] = xyz

        return poses
