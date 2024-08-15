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
import datetime
import importlib
import os
import time
from abc import ABC
from dataclasses import dataclass, field

import numpy as np
from matplotlib import pyplot as plt

# Button names
START_BUTTON = "Play [SPACE]"
PAUSE_BUTTON = "Hold [SPACE]"
NEXT_FRAME_BUTTON = "Step Frame [N]"
LOCAL_VIEW_BUTTON = "Local View [V]"
GLOBAL_VIEW_BUTTON = "Global View [V]"
CENTER_VIEWPOINT_BUTTON = "Center Viewpoint [C]"
PREV_CLOSURE_BUTTON = "Prev Closure [P]"
NEXT_CLOSURE_BUTTON = "Next Closure [N]"
ALIGN_CLOSURE_BUTTON = "Align [A]"
CLOSURES_VIEW_BUTTON = "Switch to MapClosures View [M]"
REGISTRATION_VIEW_BUTTON = "Switch to Registration View [M]"
OPEN_DENSITY_VIEW_BUTTON = "Open Density Maps Viewer [D]"
QUIT_DENSITY_VIEW_BUTTON = "Quit Density Maps Viewer [D]"
QUIT_BUTTON = "Quit [Esc]"

# Colors
BACKGROUND_COLOR = [1.0, 1.0, 1.0]
SOURCE_COLOR = [0.8470, 0.1058, 0.3764]
TARGET_COLOR = [0.0, 0.3019, 0.2509]
TRAJECTORY_COLOR = [0.1176, 0.5333, 0.8980]

# Size constants
SOURCE_PTS_SIZE = 0.06
TARGET_PTS_SIZE = 0.08

RED = np.array([0.5, 0.0, 0.0])
BLUE = np.array([0.4, 0.5, 0.9])


@dataclass
class LoopClosureData:
    num_closures: int = 0
    current_closure_id: int = 0
    closure_poses: list = field(default_factory=list)
    closure_edges: list = field(default_factory=list)
    source_local_maps: list = field(default_factory=list)
    target_local_maps: list = field(default_factory=list)
    source_density_maps: list = field(default_factory=list)
    target_density_maps: list = field(default_factory=list)


@dataclass
class VisualizerStateMachine:
    block_execution: bool = True
    play_mode: bool = False
    view_frame: bool = True
    view_local_map: bool = True
    global_view: bool = False

    view_closures: bool = False
    view_closure_query: bool = True
    view_closure_reference: bool = True
    align_closures: bool = True
    view_density_map: bool = False

    background_color: list = field(default_factory=lambda: BACKGROUND_COLOR)
    source_points_size: float = SOURCE_PTS_SIZE
    map_points_size: float = TARGET_PTS_SIZE
    query_points_size: float = SOURCE_PTS_SIZE
    reference_points_size: float = TARGET_PTS_SIZE


class StubVisualizer(ABC):
    def __init__(self):
        pass

    def pause_vis(self):
        pass

    def update_registration(self, *kwargs):
        pass

    def update_closures(self, *kwargs):
        pass


class Visualizer(StubVisualizer):
    def __init__(self):
        # Initialize GUI
        try:
            self._ps = importlib.import_module("polyscope")
            self._gui = self._ps.imgui
        except ModuleNotFoundError as err:
            print(f'polyscope is not installed on your system, run "pip install polyscope"')
            exit(1)

        # Initialize GUI States
        self._states = VisualizerStateMachine()
        self._ref_density_viewer = None
        self._query_density_viewer = None
        self.fig = None

        # Create data
        self._trajectory = []
        self._last_frame_pose = np.eye(4)
        self.loop_closures_data = LoopClosureData()
        self._last_frame_to_local_map_pose = np.eye(4)

        self._initialize_registration_visualizers()

    def update_registration(self, source, local_map, frame_pose, frame_to_local_map_pose):
        self._update_registraion(source, local_map, frame_pose, frame_to_local_map_pose)
        self._last_frame_pose = frame_pose
        self._last_frame_to_local_map_pose = frame_to_local_map_pose
        while self._states.block_execution:
            self._ps.frame_tick()
            if self._states.play_mode:
                break
        self._states.block_execution = not self._states.block_execution

    def update_closures(
        self,
        reference_local_map,
        query_local_map,
        reference_density_map,
        query_density_map,
        closure_pose,
        closure_indices,
    ):
        self.loop_closures_data.num_closures += 1
        self._update_closures(
            reference_local_map,
            query_local_map,
            reference_density_map,
            query_density_map,
            closure_pose,
            closure_indices,
        )

    def _initialize_registration_visualizers(self):
        self._ps.set_program_name("MapClosures Visualizer")
        self._ps.init()
        self._ps.set_ground_plane_mode("none")
        self._ps.set_background_color(BACKGROUND_COLOR)
        self._ps.set_verbosity(0)
        self._ps.set_user_callback(self._main_gui_callback)
        self._ps.set_build_default_gui_panels(False)

    # --- GUI Callbacks ------------------------------------------------
    def _main_gui_callback(self):
        # GUI callbacks
        if not self._states.view_closures:
            self._start_pause_callback()
            if not self._states.play_mode:
                self._gui.SameLine()
                self._next_frame_callback()
            self._gui.Separator()
            self._background_color_callback()
            self._gui.Separator()
            self._registration_controls_callback()
            self._gui.Separator()
            self._center_viewpoint_callback()
            self._gui.SameLine()
            self._global_view_callback()
        else:
            self._background_color_callback()
            self._gui.Separator()
            self._closure_query_map_callback()
            self._closure_reference_map_callback()
            self._gui.Separator()
            self._center_viewpoint_callback()
        self._gui.SameLine()
        self._quit_callback()

        if self.loop_closures_data.num_closures != 0:
            self._closure_window_callback()

    def _closure_window_callback(self):
        self._gui.Begin(
            f"Closures Window: No. of Closures: {self.loop_closures_data.num_closures}", open=True
        )
        self._switch_view_callback()
        if self._states.view_closures:
            self._gui.Separator()
            self._closure_controls_callback()
        self._gui.End()

    def _start_pause_callback(self):
        button_name = PAUSE_BUTTON if self._states.play_mode else START_BUTTON
        if self._gui.Button(button_name) or self._gui.IsKeyPressed(self._gui.ImGuiKey_Space):
            self._states.play_mode = not self._states.play_mode

    def _next_frame_callback(self):
        if self._gui.Button(NEXT_FRAME_BUTTON) or self._gui.IsKeyPressed(self._gui.ImGuiKey_N):
            self._states.block_execution = not self._states.block_execution

    def _registration_source_callback(self):
        changed, self._states.source_points_size = self._gui.SliderFloat(
            "##frame_size", self._states.source_points_size, v_min=0.01, v_max=0.6
        )
        if changed:
            self._ps.get_point_cloud("current_frame").set_radius(
                self._states.source_points_size, relative=False
            )
        self._gui.SameLine()
        changed, self._states.view_frame = self._gui.Checkbox("Frame", self._states.view_frame)
        if changed:
            self._ps.get_point_cloud("current_frame").set_enabled(self._states.view_frame)

    def _registration_localmap_callback(self):
        changed, self._states.map_points_size = self._gui.SliderFloat(
            "##map_size", self._states.map_points_size, v_min=0.01, v_max=0.6
        )
        if changed:
            self._ps.get_point_cloud("local_map").set_radius(
                self._states.map_points_size, relative=False
            )
        self._gui.SameLine()
        changed, self._states.view_local_map = self._gui.Checkbox(
            "Local Map", self._states.view_local_map
        )
        if changed:
            self._ps.get_point_cloud("local_map").set_enabled(self._states.view_local_map)

    def _registration_controls_callback(self):
        self._registration_source_callback()
        self._registration_localmap_callback()

    def _update_registraion(self, source, local_map, frame_pose, frame_to_local_map_pose):
        source_cloud = self._ps.register_point_cloud(
            "current_frame",
            source,
            color=SOURCE_COLOR,
            point_render_mode="quad",
        )
        source_cloud.set_radius(self._states.source_points_size, relative=False)
        if self._states.global_view:
            source_cloud.set_transform(frame_pose)
        else:
            source_cloud.set_transform(np.eye(4))
        source_cloud.set_enabled(self._states.view_frame)

        map_cloud = self._ps.register_point_cloud(
            "local_map",
            local_map,
            color=TARGET_COLOR,
            point_render_mode="quad",
        )
        map_cloud.set_radius(self._states.map_points_size, relative=False)
        if self._states.global_view:
            map_cloud.set_transform(frame_pose @ np.linalg.inv(frame_to_local_map_pose))
        else:
            map_cloud.set_transform(np.linalg.inv(frame_to_local_map_pose))
        map_cloud.set_enabled(self._states.view_local_map)

        # TRAJECTORY (only visible in global view)
        self._trajectory.append(frame_pose)
        if self._states.global_view:
            self._register_trajectory()

    def _register_trajectory(self):
        trajectory_cloud = self._ps.register_point_cloud(
            "trajectory",
            np.asarray(self._trajectory)[:, :3, -1],
            color=BLUE,
        )
        trajectory_cloud.set_radius(0.3, relative=False)
        if self.loop_closures_data.num_closures != 0:
            closure_lines = self._ps.register_curve_network(
                "loop closures",
                np.array(self._trajectory)[:, :3, -1],
                np.array(self.loop_closures_data.closure_edges),
                color=RED,
            )
            closure_lines.set_radius(0.1, relative=False)

    def _unregister_trajectory(self):
        self._ps.remove_point_cloud("trajectory")
        if self.loop_closures_data.num_closures != 0:
            self._ps.remove_curve_network("loop closures")

    def _switch_view_callback(self):
        if self.loop_closures_data.num_closures != 0:
            self._gui.Separator()
            BUTTON_NAME = (
                REGISTRATION_VIEW_BUTTON if self._states.view_closures else CLOSURES_VIEW_BUTTON
            )
            if self._gui.Button(BUTTON_NAME) or self._gui.IsKeyPressed(self._gui.ImGuiKey_M):
                self._states.view_closures = not self._states.view_closures
                if self._states.view_closures:
                    self._states.view_frame = False
                    self._states.view_local_map = False
                    self._states.view_closure_query = True
                    self._states.view_closure_reference = True
                    self._ps.get_point_cloud("current_frame").set_enabled(self._states.view_frame)
                    self._ps.get_point_cloud("local_map").set_enabled(self._states.view_local_map)
                    self._states.play_mode = False
                    self._render_closure(self.loop_closures_data.current_closure_id)
                    if not self._states.global_view:
                        self._register_trajectory()
                else:
                    self._states.view_frame = True
                    self._states.view_local_map = True
                    self._states.view_closure_query = False
                    self._states.view_closure_reference = False
                    self._render_closure(self.loop_closures_data.current_closure_id)
                    self._ps.get_point_cloud("current_frame").set_enabled(self._states.view_frame)
                    self._ps.get_point_cloud("local_map").set_enabled(self._states.view_local_map)
                    if not self._states.global_view:
                        self._unregister_trajectory()
                    self._ref_density_viewer = None
                    self._query_density_viewer = None
                    if self._states.view_density_map:
                        self._states.view_density_map = False
                        plt.close("all")

    def _closure_query_map_callback(self):
        changed, self._states.query_points_size = self._gui.SliderFloat(
            "##query_size", self._states.query_points_size, v_min=0.01, v_max=0.6
        )
        if changed:
            self._ps.get_point_cloud("query_map").set_radius(
                self._states.query_points_size, relative=False
            )
        self._gui.SameLine()
        changed, self._states.view_closure_query = self._gui.Checkbox(
            "Query Map", self._states.view_closure_query
        )
        if changed:
            self._ps.get_point_cloud("query_map").set_enabled(self._states.view_closure_query)

    def _closure_reference_map_callback(self):
        changed, self._states.reference_points_size = self._gui.SliderFloat(
            "##reference_size", self._states.reference_points_size, v_min=0.01, v_max=0.6
        )
        if changed:
            self._ps.get_point_cloud("reference_map").set_radius(
                self._states.reference_points_size, relative=False
            )
        self._gui.SameLine()
        changed, self._states.view_closure_reference = self._gui.Checkbox(
            "Reference Map", self._states.view_closure_reference
        )
        if changed:
            self._ps.get_point_cloud("reference_map").set_enabled(
                self._states.view_closure_reference
            )

    def _closure_alignment_callback(self):
        changed, self._states.align_closures = self._gui.Checkbox(
            ALIGN_CLOSURE_BUTTON, self._states.align_closures
        )
        if changed:
            self._render_closure(self.loop_closures_data.current_closure_id)

        if self._gui.IsKeyPressed(self._gui.ImGuiKey_A):
            self._states.align_closures = not self._states.align_closures
            self._render_closure(self.loop_closures_data.current_closure_id)

    def _closure_navigate_callback(self):
        changed, self.loop_closures_data.current_closure_id = self._gui.SliderInt(
            f"###Closure Index",
            self.loop_closures_data.current_closure_id,
            v_min=0,
            v_max=self.loop_closures_data.num_closures - 1,
            format="Closure Id: %d",
        )
        if changed:
            self._render_closure(self.loop_closures_data.current_closure_id)
            if self._states.view_density_map:
                self._ref_density_viewer.set_data(
                    self.loop_closures_data.source_density_maps[
                        self.loop_closures_data.current_closure_id
                    ]
                )
                self._query_density_viewer.set_data(
                    self.loop_closures_data.target_density_maps[
                        self.loop_closures_data.current_closure_id
                    ]
                )
                plt.gcf().canvas.draw_idle()
                plt.gcf().canvas.start_event_loop(1e-6)

    def _density_map_callback(self):
        BUTTON = (
            OPEN_DENSITY_VIEW_BUTTON
            if not self._states.view_density_map
            else QUIT_DENSITY_VIEW_BUTTON
        )
        if self._gui.Button(BUTTON) or self._gui.IsKeyPressed(self._gui.ImGuiKey_D):
            self._states.view_density_map = not self._states.view_density_map
            if self._states.view_density_map:
                self.fig = plt.figure()
                plt.ion()
                ax_ref = self.fig.add_subplot(1, 2, 1)
                self._ref_density_viewer = ax_ref.imshow(
                    self.loop_closures_data.source_density_maps[
                        self.loop_closures_data.current_closure_id
                    ],
                    cmap="gray",
                )
                ax_ref.set_title("Reference Density Map")

                ax_query = self.fig.add_subplot(1, 2, 2)
                self._query_density_viewer = ax_query.imshow(
                    self.loop_closures_data.target_density_maps[
                        self.loop_closures_data.current_closure_id
                    ],
                    cmap="gray",
                )
                ax_query.set_title("Query Density Map")
                plt.show(block=False)
                plt.gcf().canvas.draw_idle()
                plt.gcf().canvas.start_event_loop(1e-6)
            else:
                plt.close("all")

    def _previous_next_closure_callback(self):
        if self._gui.Button(PREV_CLOSURE_BUTTON) or self._gui.IsKeyPressed(self._gui.ImGuiKey_P):
            self.loop_closures_data.current_closure_id = (
                self.loop_closures_data.current_closure_id - 1
            ) % self.loop_closures_data.num_closures
            if self._states.view_density_map:
                self._ref_density_viewer.set_data(
                    self.loop_closures_data.source_density_maps[
                        self.loop_closures_data.current_closure_id
                    ]
                )
                self._query_density_viewer.set_data(
                    self.loop_closures_data.target_density_maps[
                        self.loop_closures_data.current_closure_id
                    ]
                )
                plt.gcf().canvas.draw_idle()
                plt.gcf().canvas.start_event_loop(1e-6)

        self._gui.SameLine()
        if self._gui.Button(NEXT_CLOSURE_BUTTON) or self._gui.IsKeyPressed(self._gui.ImGuiKey_N):
            self.loop_closures_data.current_closure_id = (
                self.loop_closures_data.current_closure_id + 1
            ) % self.loop_closures_data.num_closures
            if self._states.view_density_map:
                self._ref_density_viewer.set_data(
                    self.loop_closures_data.source_density_maps[
                        self.loop_closures_data.current_closure_id
                    ]
                )
                self._query_density_viewer.set_data(
                    self.loop_closures_data.target_density_maps[
                        self.loop_closures_data.current_closure_id
                    ]
                )
                plt.gcf().canvas.draw_idle()
                plt.gcf().canvas.start_event_loop(1e-6)

        self._render_closure(self.loop_closures_data.current_closure_id)

    def _closure_controls_callback(self):
        self._density_map_callback()
        self._gui.SameLine()
        self._previous_next_closure_callback()
        self._closure_alignment_callback()
        self._gui.SameLine()
        self._closure_navigate_callback()

    def _update_closures(
        self,
        reference_local_map,
        query_local_map,
        reference_density_map,
        query_density_map,
        closure_pose,
        closure_indices,
    ):
        self.loop_closures_data.source_local_maps.append(reference_local_map)
        self.loop_closures_data.target_local_maps.append(query_local_map)
        self.loop_closures_data.source_density_maps.append(reference_density_map)
        self.loop_closures_data.target_density_maps.append(query_density_map)
        self.loop_closures_data.closure_poses.append(closure_pose)
        self.loop_closures_data.closure_edges.append(closure_indices)
        self.loop_closures_data.current_closure_id = self.loop_closures_data.num_closures - 1
        if self._states.view_closures:
            self._render_closure(self.loop_closures_data.current_closure_id)

    def _render_closure(self, idx):
        ref_map_pose = self._trajectory[self.loop_closures_data.closure_edges[idx][0]]
        query_map_pose = self._trajectory[self.loop_closures_data.closure_edges[idx][1]]
        query_map = self._ps.register_point_cloud(
            "query_map",
            self.loop_closures_data.target_local_maps[idx],
            color=TARGET_COLOR,
            point_render_mode="quad",
        )
        query_map.set_radius(self._states.query_points_size, relative=False)
        query_map.set_transform(query_map_pose)
        query_map.set_enabled(self._states.view_closure_query)

        reference_map = self._ps.register_point_cloud(
            "reference_map",
            self.loop_closures_data.source_local_maps[idx],
            color=SOURCE_COLOR,
            point_render_mode="quad",
        )
        reference_map.set_radius(self._states.reference_points_size, relative=False)
        if self._states.align_closures:
            reference_map.set_transform(query_map_pose @ self.loop_closures_data.closure_poses[idx])
        else:
            reference_map.set_transform(ref_map_pose)
        reference_map.set_enabled(self._states.view_closure_reference)

    def _background_color_callback(self):
        changed, self._states.background_color = self._gui.ColorEdit3(
            "Background Color",
            self._states.background_color,
        )
        if changed:
            self._ps.set_background_color(self._states.background_color)

    def _global_view_callback(self):
        button_name = LOCAL_VIEW_BUTTON if self._states.global_view else GLOBAL_VIEW_BUTTON
        if self._gui.Button(button_name) or self._gui.IsKeyPressed(self._gui.ImGuiKey_V):
            self._states.global_view = not self._states.global_view
            if self._states.global_view:
                self._ps.get_point_cloud("current_frame").set_transform(self._last_frame_pose)
                self._ps.get_point_cloud("local_map").set_transform(
                    self._last_frame_pose @ np.linalg.inv(self._last_frame_to_local_map_pose)
                )
                self._register_trajectory()
            else:
                self._ps.get_point_cloud("current_frame").set_transform(np.eye(4))
                self._ps.get_point_cloud("local_map").set_transform(
                    np.linalg.inv(self._last_frame_to_local_map_pose)
                )
                self._unregister_trajectory()
            self._ps.reset_camera_to_home_view()

    def _center_viewpoint_callback(self):
        if self._gui.Button(CENTER_VIEWPOINT_BUTTON) or self._gui.IsKeyPressed(
            self._gui.ImGuiKey_C
        ):
            self._ps.reset_camera_to_home_view()

    def _quit_callback(self):
        posX = (
            self._gui.GetCursorPosX()
            + self._gui.GetColumnWidth()
            - self._gui.CalcTextSize(QUIT_BUTTON)[0]
            - self._gui.GetScrollX()
            - self._gui.ImGuiStyleVar_ItemSpacing
        )
        self._gui.SetCursorPosX(posX)
        if self._gui.Button(QUIT_BUTTON) or self._gui.IsKeyPressed(self._gui.ImGuiKey_Escape):
            print("Destroying Visualizer")
            self._ps.unshow()
            os._exit(0)
