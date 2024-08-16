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
import importlib
import os
from abc import ABC
from dataclasses import dataclass, field

import numpy as np
from matplotlib import pyplot as plt

# Button names
START = "Play [SPACE]"
PAUSE = "Hold [SPACE]"
NEXT_FRAME = "Step Frame [N]"
LOCAL_VIEW = "Local View [V]"
GLOBAL_VIEW = "Global View [V]"
CENTER_VIEWPOINT = "Center Viewpoint [C]"
PREV_CLOSURE = "Prev Closure [P]"
NEXT_CLOSURE = "Next Closure [N]"
ALIGN_CLOSURE = "Align [A]"
CLOSURES_VIEW = "Switch to MapClosures View [M]"
REGISTRATION_VIEW = "Switch to Registration View [M]"
OPEN_DENSITY_VIEW = "Open Density Maps Viewer [D]"
QUIT_DENSITY_VIEW = "Quit Density Maps Viewer [D]"
QUIT = "Quit [Q]"

# Colors
BACKGROUND_COLOR = [1.0, 1.0, 1.0]
SOURCE_COLOR = [0.8470, 0.1058, 0.3764]
TARGET_COLOR = [0.0, 0.3019, 0.2509]
TRAJECTORY_COLOR = [0.1176, 0.5333, 0.8980]

# Size constants
SOURCE_PTS_SIZE = 0.06
TARGET_PTS_SIZE = 0.08
I = np.eye(4)

RED = np.array([0.5, 0.0, 0.0])
BLUE = np.array([0.4, 0.5, 0.9])


@dataclass
class LoopClosureData:
    num_closures: int = 0
    current_closure_id: int = 0
    local_maps: list = field(default_factory=list)
    density_maps: list = field(default_factory=list)
    local_map_poses: list = field(default_factory=list)
    closure_edges: list = field(default_factory=list)
    closure_alignment: list = field(default_factory=list)


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

    def update_data(self, *kwargs):
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
        self._last_frame_pose = I
        self._closures_data = LoopClosureData()
        self._last_frame_to_local_map_pose = I

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

    def update_data(self, local_map, density_map, local_map_pose):
        self._closures_data.local_maps.append(local_map)
        self._closures_data.density_maps.append(density_map)
        self._closures_data.local_map_poses.append(local_map_pose)

    def update_closures(self, closure_alignment, closure_edge):
        self._closures_data.current_closure_id = self._closures_data.num_closures
        self._closures_data.closure_edges.append(closure_edge)
        self._closures_data.closure_alignment.append(closure_alignment)
        self._closures_data.num_closures += 1
        self._update_closures()

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
        if not self._states.view_closures:
            self._start_pause_callback()
            if not self._states.play_mode:
                self._gui.SameLine()
                self._next_frame_callback()
            self._gui.Separator()
            self._registration_source_callback()
            self._registration_localmap_callback()
        else:
            self._previous_next_closure_callback()
            self._gui.SameLine()
            self._density_map_callback()
            self._gui.Separator()
            self._closure_query_map_callback()
            self._closure_reference_map_callback()
            self._gui.Separator()
            self._closure_alignment_callback()
            self._gui.SameLine()
            self._closure_navigate_callback()
            if self._states.view_density_map:
                plt.gcf().canvas.draw()
                plt.gcf().canvas.start_event_loop(1e-6)
        self._gui.Separator()
        self._background_color_callback()
        self._gui.Separator()
        self._global_view_callback()
        self._gui.SameLine()
        self._center_viewpoint_callback()
        self._gui.SameLine()
        self._quit_callback()
        if self._closures_data.num_closures:
            self._closure_window_callback()

    def _closure_window_callback(self):
        window_pos = self._gui.GetWindowPos()
        window_width = self._gui.GetWindowWidth()
        window_height = self._gui.GetWindowHeight()
        self._gui.Begin(f"No. of Closures: {self._closures_data.num_closures}", open=True)
        self._gui.SetWindowPos((window_pos[0], window_pos[1] + window_height + 20))
        self._gui.SetWindowSize((window_width, 4 * self._gui.GetTextLineHeight()))
        self._switch_view_callback()
        self._gui.End()

    def _start_pause_callback(self):
        button_name = PAUSE if self._states.play_mode else START
        if self._gui.Button(button_name) or self._gui.IsKeyPressed(self._gui.ImGuiKey_Space):
            self._states.play_mode = not self._states.play_mode

    def _next_frame_callback(self):
        if self._gui.Button(NEXT_FRAME) or self._gui.IsKeyPressed(self._gui.ImGuiKey_N):
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

    def _update_registraion(self, source, local_map, frame_pose, frame_to_local_map_pose):
        source_cloud = self._ps.register_point_cloud(
            "current_frame",
            source,
            color=SOURCE_COLOR,
            point_render_mode="quad",
        )
        source_cloud.set_radius(self._states.source_points_size, relative=False)
        map_cloud = self._ps.register_point_cloud(
            "local_map",
            local_map,
            color=TARGET_COLOR,
            point_render_mode="quad",
        )
        map_cloud.set_radius(self._states.map_points_size, relative=False)
        if self._states.global_view:
            source_cloud.set_transform(frame_pose)
            map_cloud.set_transform(frame_pose @ np.linalg.inv(frame_to_local_map_pose))
        else:
            source_cloud.set_transform(I)
            map_cloud.set_transform(np.linalg.inv(frame_to_local_map_pose))
        source_cloud.set_enabled(self._states.view_frame)
        map_cloud.set_enabled(self._states.view_local_map)

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
        if self._closures_data.num_closures:
            closure_lines = self._ps.register_curve_network(
                "loop closures",
                np.array(self._closures_data.local_map_poses)[:, :3, -1],
                np.array(self._closures_data.closure_edges),
                color=RED,
            )
            closure_lines.set_radius(0.1, relative=False)

    def _unregister_trajectory(self):
        self._ps.remove_point_cloud("trajectory")
        if self._closures_data.num_closures:
            self._ps.remove_curve_network("loop closures")

    def _switch_view_callback(self):
        BUTTON_NAME = REGISTRATION_VIEW if self._states.view_closures else CLOSURES_VIEW
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
                self._render_closure()
                if self._states.view_density_map:
                    plt.gcf().canvas.draw()
                    plt.gcf().canvas.start_event_loop(1e-6)
            else:
                self._states.view_frame = True
                self._states.view_local_map = True
                self._states.view_closure_query = False
                self._states.view_closure_reference = False
                self._render_closure()
                self._ps.get_point_cloud("current_frame").set_enabled(self._states.view_frame)
                self._ps.get_point_cloud("local_map").set_enabled(self._states.view_local_map)
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
            ALIGN_CLOSURE, self._states.align_closures
        )
        if changed:
            self._render_closure()
        if self._gui.IsKeyPressed(self._gui.ImGuiKey_A):
            self._states.align_closures = not self._states.align_closures
            self._render_closure()

    def _closure_navigate_callback(self):
        changed, self._closures_data.current_closure_id = self._gui.SliderInt(
            f"###Closure Index",
            self._closures_data.current_closure_id,
            v_min=0,
            v_max=self._closures_data.num_closures - 1,
            format="Closure Id: %d",
        )
        if changed:
            self._render_closure()

    def _previous_next_closure_callback(self):
        if self._gui.Button(PREV_CLOSURE) or self._gui.IsKeyPressed(self._gui.ImGuiKey_P):
            self._closures_data.current_closure_id = (
                self._closures_data.current_closure_id - 1
            ) % self._closures_data.num_closures
        self._gui.SameLine()
        if self._gui.Button(NEXT_CLOSURE) or self._gui.IsKeyPressed(self._gui.ImGuiKey_N):
            self._closures_data.current_closure_id = (
                self._closures_data.current_closure_id + 1
            ) % self._closures_data.num_closures
        self._render_closure()

    def _density_map_callback(self):
        BUTTON = OPEN_DENSITY_VIEW if not self._states.view_density_map else QUIT_DENSITY_VIEW
        if self._gui.Button(BUTTON) or self._gui.IsKeyPressed(self._gui.ImGuiKey_D):
            self._states.view_density_map = not self._states.view_density_map
            if self._states.view_density_map:
                id = self._closures_data.current_closure_id
                [ref_id, query_id] = self._closures_data.closure_edges[id]
                plt.ion()
                self.fig = plt.figure()
                plt.show(block=False)
                ax_ref = self.fig.add_subplot(1, 2, 1)
                ax_ref.set_title("Reference Density Map")
                ax_query = self.fig.add_subplot(1, 2, 2)
                ax_query.set_title("Query Density Map")
                self._ref_density_viewer = ax_ref.imshow(
                    self._closures_data.density_maps[ref_id], cmap="gray"
                )
                self._query_density_viewer = ax_query.imshow(
                    self._closures_data.density_maps[query_id], cmap="gray"
                )
                plt.gcf().canvas.draw()
                plt.gcf().canvas.start_event_loop(1e-6)
            else:
                plt.close("all")

    def _update_closures(self):
        if self._states.view_closures:
            self._render_closure()

    def _render_closure(self):
        id = self._closures_data.current_closure_id
        [ref_id, query_id] = self._closures_data.closure_edges[id]
        ref_map_pose = self._closures_data.local_map_poses[ref_id]
        query_map_pose = self._closures_data.local_map_poses[query_id]
        query_map = self._ps.register_point_cloud(
            "query_map",
            self._closures_data.local_maps[query_id],
            color=TARGET_COLOR,
            point_render_mode="quad",
        )
        query_map.set_radius(self._states.query_points_size, relative=False)
        reference_map = self._ps.register_point_cloud(
            "reference_map",
            self._closures_data.local_maps[ref_id],
            color=SOURCE_COLOR,
            point_render_mode="quad",
        )
        reference_map.set_radius(self._states.reference_points_size, relative=False)

        if self._states.global_view:
            query_map.set_transform(query_map_pose)
            if self._states.align_closures:
                reference_map.set_transform(
                    query_map_pose @ self._closures_data.closure_alignment[id]
                )
            else:
                reference_map.set_transform(ref_map_pose)
        else:
            query_map.set_transform(I)
            if self._states.align_closures:
                reference_map.set_transform(self._closures_data.closure_alignment[id])
            else:
                reference_map.set_transform(I)
        query_map.set_enabled(self._states.view_closure_query)
        reference_map.set_enabled(self._states.view_closure_reference)

        if self._states.view_density_map:
            self._ref_density_viewer.set_data(self._closures_data.density_maps[ref_id])
            self._query_density_viewer.set_data(self._closures_data.density_maps[query_id])
            plt.gcf().canvas.draw()
            plt.gcf().canvas.start_event_loop(1e-6)

    def _background_color_callback(self):
        changed, self._states.background_color = self._gui.ColorEdit3(
            "Background Color",
            self._states.background_color,
        )
        if changed:
            self._ps.set_background_color(self._states.background_color)

    def _global_view_callback(self):
        button_name = LOCAL_VIEW if self._states.global_view else GLOBAL_VIEW
        if self._gui.Button(button_name) or self._gui.IsKeyPressed(self._gui.ImGuiKey_V):
            self._states.global_view = not self._states.global_view
            if self._states.global_view:
                self._ps.get_point_cloud("current_frame").set_transform(self._last_frame_pose)
                self._ps.get_point_cloud("local_map").set_transform(
                    self._last_frame_pose @ np.linalg.inv(self._last_frame_to_local_map_pose)
                )
                self._register_trajectory()
            else:
                self._ps.get_point_cloud("current_frame").set_transform(I)
                self._ps.get_point_cloud("local_map").set_transform(
                    np.linalg.inv(self._last_frame_to_local_map_pose)
                )
                self._unregister_trajectory()
            if self._states.view_closures:
                self._render_closure()
            self._ps.reset_camera_to_home_view()

    def _center_viewpoint_callback(self):
        if self._gui.Button(CENTER_VIEWPOINT) or self._gui.IsKeyPressed(self._gui.ImGuiKey_C):
            self._ps.reset_camera_to_home_view()

    def _quit_callback(self):
        posX = (
            self._gui.GetCursorPosX()
            + self._gui.GetColumnWidth()
            - self._gui.CalcTextSize(QUIT)[0]
            - self._gui.GetScrollX()
            - self._gui.ImGuiStyleVar_ItemSpacing
        )
        self._gui.SetCursorPosX(posX)
        if self._gui.Button(QUIT) or self._gui.IsKeyPressed(self._gui.ImGuiKey_Q):
            print("Destroying Visualizer")
            self._ps.unshow()
            os._exit(0)
