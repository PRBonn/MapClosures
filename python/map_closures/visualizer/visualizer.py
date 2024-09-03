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

from map_closures.visualizer.closures_visualizer import ClosuresVisualizer
from map_closures.visualizer.local_maps_visualizer import LocalMapVisualizer
from map_closures.visualizer.registration_visualizer import RegistrationVisualizer

# Button names
LOCAL_VIEW = "Local View [V]"
GLOBAL_VIEW = "Global View [V]"
CENTER_VIEWPOINT = "Center Viewpoint [C]"
LOCALMAP_VIEW = "Switch to LocalMap View [L]"
REGISTRATION_VIEW_1 = "Switch to Registration View [L]"
CLOSURES_VIEW = "Switch to MapClosures View [M]"
REGISTRATION_VIEW_2 = "Switch to Registration View [M]"
QUIT = "Quit [Q]"

BACKGROUND_COLOR = [1.0, 1.0, 1.0]
I = np.eye(4)


@dataclass
class LocalMapData:
    size: int = 0
    local_maps: list = field(default_factory=list)
    density_maps: list = field(default_factory=list)
    local_map_poses: list = field(default_factory=list)


class StubVisualizer(ABC):
    def __init__(self):
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

        self.localmap_data = LocalMapData()
        self.registration = RegistrationVisualizer(self._ps, self._gui)
        self.closures = ClosuresVisualizer(self._ps, self._gui, self.localmap_data)
        self.local_maps = LocalMapVisualizer(self._ps, self._gui, self.localmap_data)
        self.background_color = BACKGROUND_COLOR
        self.global_view = False

        self._initialize_visualizers()

    def update_registration(self, source, local_map, frame_pose):
        self.registration.update(source, local_map, frame_pose)

    def update_data(self, local_map, density_map, local_map_pose):
        self.localmap_data.size += 1
        self.localmap_data.local_maps.append(local_map)
        self.localmap_data.density_maps.append(density_map)
        self.localmap_data.local_map_poses.append(local_map_pose)

    def update_closures(self, alignment_pose, closure_edge):
        self.closures.update_closures(alignment_pose, closure_edge)

    def _initialize_visualizers(self):
        self._ps.set_program_name("MapClosures Visualizer")
        self._ps.init()
        self._ps.set_ground_plane_mode("none")
        self._ps.set_background_color(BACKGROUND_COLOR)
        self._ps.set_verbosity(0)
        self._ps.set_user_callback(self._main_gui_callback)
        self._ps.set_build_default_gui_panels(False)

    # --- GUI Callbacks ------------------------------------------------
    def _main_gui_callback(self):
        if not self.closures.states.toggle_view and not self.local_maps.toggle_view:
            self.registration._start_pause_callback()
            if not self.registration.states.play_mode:
                self._gui.SameLine()
                self.registration._next_frame_callback()
            self._gui.Separator()
            self.registration._source_callback()
            self.registration._localmap_callback()
        elif self.closures.states.toggle_view:
            self.closures._previous_next_closure_callback()
            self._gui.SameLine()
            self.closures._density_map_callback()
            self._gui.Separator()
            self.closures._query_map_callback()
            self.closures._reference_map_callback()
            self._gui.Separator()
            self.closures._alignment_callback()
            self._gui.SameLine()
            self.closures._closure_slider_callback()
            if self.closures.states.view_density_map:
                self.closures.matplotlib_eventloop()
        elif self.local_maps.toggle_view:
            self.local_maps._previous_next_localmap_callback()
            self._gui.SameLine()
            self.local_maps._density_map_callback()
            self._gui.Separator()
            self.local_maps._pts_size_callback()
            self.local_maps._navigate_callback()
            if self.local_maps.view_density_map:
                self.local_maps.matplotlib_eventloop()
        self._gui.Separator()
        self._background_color_callback()
        self._gui.Separator()
        self._global_view_callback()
        self._gui.SameLine()
        self._center_viewpoint_callback()
        self._gui.SameLine()
        self._quit_callback()

        if self.localmap_data.size and not self.closures.states.toggle_view:
            self._local_map_window_callback()
        if self.closures.data.size and not self.local_maps.toggle_view:
            self._closure_window_callback()

    def _local_map_window_callback(self):
        window_pos = self._gui.GetWindowPos()
        window_width = self._gui.GetWindowWidth()
        window_height = self._gui.GetWindowHeight()
        self._gui.Begin(f"No. of LocalMaps: {self.localmap_data.size}", open=True)
        self._gui.SetWindowPos((window_pos[0], window_pos[1] + window_height + 20))
        self._gui.SetWindowSize((window_width, 4 * self._gui.GetTextLineHeight()))
        self._switch_to_localmap_view_callback()
        self._gui.End()

    def _closure_window_callback(self):
        if not self.closures.states.toggle_view:
            window_pos = self._gui.GetWindowPos()
            window_width = self._gui.GetWindowWidth()
            window_height = self._gui.GetWindowHeight()
            self._gui.Begin(f"No. of Closures: {self.closures.data.size}", open=True)
            self._gui.SetWindowPos(
                (
                    window_pos[0],
                    window_pos[1] + window_height + 4 * self._gui.GetTextLineHeight() + 40,
                )
            )
            self._gui.SetWindowSize((window_width, 4 * self._gui.GetTextLineHeight()))
            self._switch_to_closures_view_callback()
            self._gui.End()
        else:
            window_pos = self._gui.GetWindowPos()
            window_width = self._gui.GetWindowWidth()
            window_height = self._gui.GetWindowHeight()
            self._gui.Begin(f"No. of Closures: {self.closures.data.size}", open=True)
            self._gui.SetWindowPos((window_pos[0], window_pos[1] + window_height + 20))
            self._gui.SetWindowSize((window_width, 4 * self._gui.GetTextLineHeight()))
            self._switch_to_closures_view_callback()
            self._gui.End()

    def _switch_to_localmap_view_callback(self):
        BUTTON_NAME = REGISTRATION_VIEW_1 if self.local_maps.toggle_view else LOCALMAP_VIEW
        if self._gui.Button(BUTTON_NAME) or self._gui.IsKeyPressed(self._gui.ImGuiKey_L):
            self.local_maps.toggle_view = not self.local_maps.toggle_view
            if self.local_maps.toggle_view:
                self.registration.states.view_frame = False
                self.registration.states.view_local_map = False
                self.local_maps.view_local_map = True
                self._ps.get_point_cloud("source").set_enabled(self.registration.states.view_frame)
                self._ps.get_point_cloud("target").set_enabled(
                    self.registration.states.view_local_map
                )
                self.registration.states.play_mode = False
                self.local_maps._update_callback()
                if self.local_maps.view_density_map:
                    self.local_maps.matplotlib_eventloop()
            else:
                self.registration.states.view_frame = True
                self.registration.states.view_local_map = True
                self.local_maps.view_local_map = False
                self.local_maps._update_callback()
                self._ps.get_point_cloud("source").set_enabled(self.registration.states.view_frame)
                self._ps.get_point_cloud("target").set_enabled(
                    self.registration.states.view_local_map
                )
                if self.local_maps.view_density_map:
                    self.local_maps.view_density_map = False
                    self.local_maps.close_density_map_fig()

    def _switch_to_closures_view_callback(self):
        BUTTON_NAME = REGISTRATION_VIEW_2 if self.closures.states.toggle_view else CLOSURES_VIEW
        if self._gui.Button(BUTTON_NAME) or self._gui.IsKeyPressed(self._gui.ImGuiKey_M):
            self.closures.states.toggle_view = not self.closures.states.toggle_view
            if self.closures.states.toggle_view:
                self.registration.states.view_frame = False
                self.registration.states.view_local_map = False
                self.closures.states.view_query = True
                self.closures.states.view_reference = True
                self._ps.get_point_cloud("source").set_enabled(self.registration.states.view_frame)
                self._ps.get_point_cloud("target").set_enabled(
                    self.registration.states.view_local_map
                )
                self.registration.states.play_mode = False
                self.closures._render_closure()
                if self.closures.states.view_density_map:
                    self.closures.matplotlib_eventloop()
            else:
                self.registration.states.view_frame = True
                self.registration.states.view_local_map = True
                self.closures.states.view_query = False
                self.closures.states.view_reference = False
                self.closures._render_closure()
                self._ps.get_point_cloud("source").set_enabled(self.registration.states.view_frame)
                self._ps.get_point_cloud("target").set_enabled(
                    self.registration.states.view_local_map
                )
                if self.closures.states.view_density_map:
                    self.closures.states.view_density_map = False
                    self.closures.close_density_map_fig()

    def _register_trajectory(self):
        self.registration._register_trajectory()
        if self.closures.data.size:
            self.closures._register_trajectory()

    def _unregister_trajectory(self):
        self.registration._unregister_trajectory()
        if self.closures.data.size:
            self.closures._unregister_trajectory()

    def _global_view_callback(self):
        button_name = LOCAL_VIEW if self.global_view else GLOBAL_VIEW
        if self._gui.Button(button_name) or self._gui.IsKeyPressed(self._gui.ImGuiKey_V):
            self.global_view = not self.global_view
            self.registration.states.global_view = self.global_view
            self.closures.states.global_view = self.global_view
            self.local_maps.global_view = self.global_view
            if self.global_view:
                self._ps.get_point_cloud("source").set_transform(self.registration.last_frame_pose)
                self._ps.get_point_cloud("target").set_transform(I)
                self._register_trajectory()
            else:
                self._ps.get_point_cloud("source").set_transform(I)
                self._ps.get_point_cloud("target").set_transform(
                    np.linalg.inv(self.registration.last_frame_pose)
                )
                self._unregister_trajectory()
            if self.closures.states.toggle_view:
                self.closures._render_closure()
            self._ps.reset_camera_to_home_view()

    def _background_color_callback(self):
        changed, self.background_color = self._gui.ColorEdit3(
            "Background Color",
            self.background_color,
        )
        if changed:
            self._ps.set_background_color(self.background_color)

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
            if self.closures.states.view_density_map:
                self.closures.close_density_map_fig()
            if self.local_maps.view_density_map:
                self.local_maps.close_density_map_fig()
            self._ps.unshow()
            os._exit(0)
