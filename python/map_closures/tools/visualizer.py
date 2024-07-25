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
import copy
import importlib
import os
from abc import ABC
from functools import partial
from typing import Callable, List

import numpy as np

CYAN = np.array([0.24, 0.898, 1.0])
GREEN = np.array([0.0, 1.0, 0.0])
RED = np.array([0.5, 0.0, 0.0])
BLACK = np.array([0.0, 0.0, 0.0])
BLUE = np.array([0.4, 0.5, 0.9])
GRAY = np.array([0.4, 0.4, 0.4])
WHITE = np.array([1.0, 1.0, 1.0])
SPHERE_SIZE = 0.20


@dataclass
class RegistrationData:
    source: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
    keypoints: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
    target: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
    frames: list = field(default_factory=list)


@dataclass
class LoopClosureData:
    loop_closures: o3d.geometry.LineSet = o3d.geometry.LineSet()
    closure_points: list = field(default_factory=list)
    closure_lines: list = field(default_factory=list)

    closures_exist: bool = False

    sources: list = field(default_factory=list)
    targets: list = field(default_factory=list)
    closure_poses: list = field(default_factory=list)

    current_closure_id: int = 0
    current_source: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
    current_target: o3d.geometry.PointCloud = o3d.geometry.PointCloud()


class StubVisualizer(ABC):
    def __init__(self):
        pass

    def pause_vis():
        pass

    def update_registration(self, source, keypoints, target_map, frame_pose, map_pose):
        pass

    def update_closures(self, source, target, closure_pose, closure_indices):
        pass


class Visualizer(StubVisualizer):
    # Public Interaface ----------------------------------------------------------------------------
    def __init__(self):
        try:
            self.o3d = importlib.import_module("open3d")
        except ModuleNotFoundError:
            print(f'open3d is not installed on your system, run "pip install open3d"')
            exit(1)

        # Initialize GUI controls
        self.block_vis = True
        self.play_crun = False
        self.reset_bounding_box = True

        # Create registration data
        self.registration_data = RegistrationData()
        self.loop_closures_data = LoopClosureData()

        # Initialize visualizer
        self.registration_vis = self.o3d.visualization.VisualizerWithKeyCallback()
        self.closure_vis = self.o3d.visualization.VisualizerWithKeyCallback()
        self._registration_key_callbacks()
        self._initialize_registration_visualizers()

        # Visualization options
        self.render_map = False
        self.render_source = True
        self.render_keypoints = False
        self.global_view = True
        self.render_trajectory = True

        # Cache the state of the visualizer
        self.state = (
            self.render_map,
            self.render_keypoints,
            self.render_source,
        )

    def pause_vis(self):
        self.play_crun = False
        while True:
            self.registration_vis.poll_events()
            self.registration_vis.update_renderer()
            self.closure_vis.poll_events()
            self.closure_vis.update_renderer()
            if self.play_crun:
                break

    def update_registration(self, source, keypoints, target_map, frame_pose, map_pose):
        target = target_map.point_cloud()
        self._update_registraion(source, keypoints, target, frame_pose, map_pose)
        while self.block_vis:
            self.registration_vis.poll_events()
            self.registration_vis.update_renderer()
            self.closure_vis.poll_events()
            self.closure_vis.update_renderer()
            if self.play_crun:
                break
        self.block_vis = not self.block_vis

    def update_closures(self, source, target, closure_pose, closure_indices):
        self._update_closures(source, target, closure_pose, closure_indices)
        if self.loop_closures_data.closures_exist == False:
            self._initialize_closure_visualizers()
            self._loopclosure_key_callbacks()
            self.loop_closures_data.closures_exist = True

        # Render trajectory, only if it make sense (global view)
        if self.render_trajectory and self.global_view:
            self.registration_vis.update_geometry(self.loop_closures_data.loop_closures)

    # Private Interaface ---------------------------------------------------------------------------
    def _initialize_registration_visualizers(self):
        w_name = "Registration Visualizer"
        self.registration_vis.create_window(window_name=w_name, width=1920, height=1080)
        self.registration_vis.add_geometry(self.registration_data.source, reset_bounding_box=False)
        self._set_black_background(self.registration_vis)
        self.registration_vis.get_render_option().point_size = 1
        print(
            f"{w_name} initialized. Press:\n"
            "\t[SPACE] to pause/start\n"
            "\t  [ESC] to exit\n"
            "\t    [N] to step\n"
            "\t    [F] to toggle on/off the input cloud to the pipeline\n"
            "\t    [K] to toggle on/off the subsbampled frame\n"
            "\t    [M] to toggle on/off the local map\n"
            "\t    [V] to toggle ego/global viewpoint\n"
            "\t    [T] to toggle the trajectory view(only available in global view)\n"
            "\t    [C] to center the viewpoint\n"
            "\t    [W] to toggle a white background\n"
            "\t    [B] to toggle a black background\n"
        )

    def _initialize_closure_visualizers(self):
        w_name = "Loop Closure Visualizer"
        self.closure_vis.create_window(window_name=w_name, width=1920, height=1080)
        self.closure_vis.add_geometry(
            self.loop_closures_data.current_source, reset_bounding_box=False
        )
        self.closure_vis.add_geometry(
            self.loop_closures_data.current_target, reset_bounding_box=False
        )
        self.registration_vis.add_geometry(
            self.loop_closures_data.loop_closures, reset_bounding_box=False
        )
        self._set_black_background(self.closure_vis)
        self.closure_vis.get_render_option().point_size = 1
        print(
            f"{w_name} initialized. Press:\n"
            "\t[SPACE] to pause/start\n"
            "\t    [P] to step to previous closure\n"
            "\t    [N] to step to next closure\n"
            "\t    [C] to center the viewpoint\n"
            "\t    [W] to toggle a white background\n"
            "\t    [B] to toggle a black background\n"
        )

    def _registration_key_callback(self, keys: List, callback: Callable):
        for key in keys:
            self.registration_vis.register_key_callback(ord(str(key)), partial(callback))

    def _loopclosure_key_callback(self, keys: List, callback: Callable):
        for key in keys:
            self.closure_vis.register_key_callback(ord(str(key)), partial(callback))

    def _registration_key_callbacks(self):
        self._registration_key_callback(["Ā", "Q", "\x1b"], self._quit)
        self._registration_key_callback([" "], self._start_stop)
        self._registration_key_callback(["N"], self._next_frame)
        self._registration_key_callback(["V"], self._toggle_view)
        self._registration_key_callback(["C"], self._center_viewpoint)
        self._registration_key_callback(["F"], self._toggle_source)
        self._registration_key_callback(["K"], self._toggle_keypoints)
        self._registration_key_callback(["M"], self._toggle_map)
        self._registration_key_callback(["T"], self._toggle_trajectory)
        self._registration_key_callback(["B"], self._set_black_background)
        self._registration_key_callback(["W"], self._set_white_background)

    def _loopclosure_key_callbacks(self):
        self._loopclosure_key_callback([" "], self._start_stop)
        self._loopclosure_key_callback(["P"], self._prev_closure)
        self._loopclosure_key_callback(["N"], self._next_closure)
        self._loopclosure_key_callback(["C"], self._center_viewpoint)
        self._loopclosure_key_callback(["B"], self._set_black_background)
        self._loopclosure_key_callback(["W"], self._set_white_background)

    def _set_black_background(self, vis):
        vis.get_render_option().background_color = [0.0, 0.0, 0.0]

    def _set_white_background(self, vis):
        vis.get_render_option().background_color = [1.0, 1.0, 1.0]

    def _quit(self, vis):
        print("Destroying Visualizer")
        vis.destroy_window()
        os._exit(0)

    def _next_frame(self, vis):
        self.block_vis = not self.block_vis

    def _prev_closure(self, vis):
        self.loop_closures_data.current_closure_id = (
            self.loop_closures_data.current_closure_id - 1
        ) % len(self.loop_closures_data.closure_poses)
        self._update_closure(self.loop_closures_data.current_closure_id)

    def _next_closure(self, vis):
        self.loop_closures_data.current_closure_id = (
            self.loop_closures_data.current_closure_id + 1
        ) % len(self.loop_closures_data.closure_poses)
        self._update_closure(self.loop_closures_data.current_closure_id)

    def _start_stop(self, vis):
        self.play_crun = not self.play_crun

    def _toggle_source(self, vis):
        if self.render_keypoints:
            self.render_keypoints = False
            self.render_source = True
            self.registration_vis.remove_geometry(
                self.registration_data.keypoints, reset_bounding_box=False
            )
        else:
            self.render_source = not self.render_source

        if self.render_source:
            self.registration_vis.add_geometry(
                self.registration_data.source, reset_bounding_box=False
            )
        else:
            self.registration_vis.remove_geometry(
                self.registration_data.source, reset_bounding_box=False
            )
        return False

    def _toggle_keypoints(self, vis):
        if self.render_source:
            self.render_source = False
            self.render_keypoints = True
            self.registration_vis.remove_geometry(
                self.registration_data.source, reset_bounding_box=False
            )
        else:
            self.render_keypoints = not self.render_keypoints

        if self.render_keypoints:
            self.registration_vis.add_geometry(
                self.registration_data.keypoints, reset_bounding_box=False
            )
        else:
            self.registration_vis.remove_geometry(
                self.registration_data.keypoints, reset_bounding_box=False
            )
        return False

    def _toggle_map(self, vis):
        self.render_map = not self.render_map
        if self.render_map:
            self.registration_vis.add_geometry(
                self.registration_data.target, reset_bounding_box=False
            )
        else:
            self.registration_vis.remove_geometry(
                self.registration_data.target, reset_bounding_box=False
            )
        return False

    def _toggle_view(self, vis):
        self.global_view = not self.global_view
        self._trajectory_handle()

    def _center_viewpoint(self, vis):
        vis.reset_view_point(True)

    def _toggle_trajectory(self, vis):
        if not self.global_view:
            return False
        self.render_trajectory = not self.render_trajectory
        self._trajectory_handle()
        return False

    def _trajectory_handle(self):
        if self.render_trajectory and self.global_view:
            for frame in self.registration_data.frames:
                self.registration_vis.add_geometry(frame, reset_bounding_box=False)
            if self.loop_closures_data.closures_exist:
                self.registration_vis.add_geometry(
                    self.loop_closures_data.loop_closures, reset_bounding_box=False
                )
        else:
            for frame in self.registration_data.frames:
                self.registration_vis.remove_geometry(frame, reset_bounding_box=False)
            if self.loop_closures_data.closures_exist:
                self.registration_vis.remove_geometry(
                    self.loop_closures_data.loop_closures, reset_bounding_box=False
                )

    def _update_registraion(self, source, keypoints, target, pose, frame_to_map_pose):
        # Source hot frame
        if self.render_source:
            self.registration_data.source.points = self.o3d.utility.Vector3dVector(source)
            self.registration_data.source.paint_uniform_color(CYAN)
            if self.global_view:
                self.registration_data.source.transform(pose)
            self.registration_vis.update_geometry(self.registration_data.source)

        # Keypoints
        if self.render_keypoints:
            self.registration_data.keypoints.points = self.o3d.utility.Vector3dVector(keypoints)
            self.registration_data.keypoints.paint_uniform_color(CYAN)
            if self.global_view:
                self.registration_data.keypoints.transform(pose)
            self.registration_vis.update_geometry(self.registration_data.keypoints)

        # Target Map
        if self.render_map:
            target = copy.deepcopy(target)
            self.registration_data.target.points = self.o3d.utility.Vector3dVector(target)
            if self.global_view:
                self.registration_data.target.paint_uniform_color(GRAY)
                self.registration_data.target.transform(pose @ np.linalg.inv(frame_to_map_pose))
            else:
                self.registration_data.target.transform(np.linalg.inv(frame_to_map_pose))
            self.registration_vis.update_geometry(self.registration_data.target)

        # Update always a list with all the trajectories
        new_frame = self.o3d.geometry.TriangleMesh.create_sphere(SPHERE_SIZE)
        new_frame.paint_uniform_color(BLUE)
        new_frame.compute_vertex_normals()
        new_frame.transform(pose)
        self.registration_data.frames.append(new_frame)
        self.loop_closures_data.closure_points.append(pose[:3, -1])

        # Render trajectory, only if it make sense (global view)
        if self.render_trajectory and self.global_view:
            self.registration_vis.add_geometry(
                self.registration_data.frames[-1], reset_bounding_box=False
            )

        if self.reset_bounding_box:
            self.registration_vis.reset_view_point(True)
            self.reset_bounding_box = False

    def _update_closures(self, source, target, closure_pose, closure_indices):
        self.loop_closures_data.sources.append(self.o3d.utility.Vector3dVector(source))
        self.loop_closures_data.targets.append(self.o3d.utility.Vector3dVector(target))
        self.loop_closures_data.closure_poses.append(closure_pose)
        self._update_closure(len(self.loop_closures_data.closure_poses) - 1)

        self.loop_closures_data.closure_lines.append(closure_indices)
        self.loop_closures_data.loop_closures.points = self.o3d.utility.Vector3dVector(
            self.loop_closures_data.closure_points
        )
        self.loop_closures_data.loop_closures.lines = self.o3d.utility.Vector2iVector(
            self.loop_closures_data.closure_lines
        )
        self.loop_closures_data.loop_closures.paint_uniform_color(RED)

    def _update_closure(self, idx):
        self.loop_closures_data.current_source.points = self.loop_closures_data.sources[idx]
        self.loop_closures_data.current_source.transform(self.loop_closures_data.closure_poses[idx])
        self.loop_closures_data.current_source.paint_uniform_color(RED)
        self.loop_closures_data.current_target.points = self.loop_closures_data.targets[idx]
        self.loop_closures_data.current_target.paint_uniform_color(GREEN)

        self.closure_vis.update_geometry(self.loop_closures_data.current_source)
        self.closure_vis.update_geometry(self.loop_closures_data.current_target)
