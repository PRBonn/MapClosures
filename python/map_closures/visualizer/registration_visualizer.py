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
from dataclasses import dataclass, field

import numpy as np

# Button names
START = "Play [SPACE]"
PAUSE = "Hold [SPACE]"
NEXT_FRAME = "Step Frame [N]"

# Colors
SOURCE_COLOR = [0.8470, 0.1058, 0.3764]
TARGET_COLOR = [0.0, 0.3019, 0.2509]
TRAJECTORY_COLOR = [0.1176, 0.5333, 0.8980]

# Size constants
SOURCE_PTS_SIZE = 0.06
TARGET_PTS_SIZE = 0.08
I = np.eye(4)


@dataclass
class RegsitrationStateMachine:
    block_execution: bool = True
    play_mode: bool = False
    view_frame: bool = True
    view_local_map: bool = True
    global_view: bool = False

    source_points_size: float = SOURCE_PTS_SIZE
    map_points_size: float = TARGET_PTS_SIZE


class RegistrationVisualizer:
    def __init__(self, ps, gui):
        self._ps = ps
        self._gui = gui

        # Initialize GUI States
        self.states = RegsitrationStateMachine()

        # Create data
        self.trajectory = []
        self.last_frame_pose = I

    def update(self, source, local_map, frame_pose):
        self._update(source, local_map, frame_pose)
        self.last_frame_pose = frame_pose
        while self.states.block_execution:
            self._ps.frame_tick()
            if self.states.play_mode:
                break
        self.states.block_execution = not self.states.block_execution

    def _start_pause_callback(self):
        button_name = PAUSE if self.states.play_mode else START
        if self._gui.Button(button_name) or self._gui.IsKeyPressed(self._gui.ImGuiKey_Space):
            self.states.play_mode = not self.states.play_mode

    def _next_frame_callback(self):
        if self._gui.Button(NEXT_FRAME) or self._gui.IsKeyPressed(self._gui.ImGuiKey_N):
            self.states.block_execution = not self.states.block_execution

    def _source_callback(self):
        changed, self.states.source_points_size = self._gui.SliderFloat(
            "##frame_size", self.states.source_points_size, v_min=0.01, v_max=0.6
        )
        if changed:
            self._ps.get_point_cloud("source").set_radius(
                self.states.source_points_size, relative=False
            )
        self._gui.SameLine()
        changed, self.states.view_frame = self._gui.Checkbox("Frame", self.states.view_frame)
        if changed:
            self._ps.get_point_cloud("source").set_enabled(self.states.view_frame)

    def _localmap_callback(self):
        changed, self.states.map_points_size = self._gui.SliderFloat(
            "##map_size", self.states.map_points_size, v_min=0.01, v_max=0.6
        )
        if changed:
            self._ps.get_point_cloud("target").set_radius(
                self.states.map_points_size, relative=False
            )
        self._gui.SameLine()
        changed, self.states.view_local_map = self._gui.Checkbox(
            "Local Map", self.states.view_local_map
        )
        if changed:
            self._ps.get_point_cloud("target").set_enabled(self.states.view_local_map)

    def _update(self, source, local_map, frame_pose):
        source_cloud = self._ps.register_point_cloud(
            "source",
            source,
            color=SOURCE_COLOR,
            point_render_mode="quad",
        )
        source_cloud.set_radius(self.states.source_points_size, relative=False)
        map_cloud = self._ps.register_point_cloud(
            "target",
            local_map,
            color=TARGET_COLOR,
            point_render_mode="quad",
        )
        map_cloud.set_radius(self.states.map_points_size, relative=False)
        if self.states.global_view:
            source_cloud.set_transform(frame_pose)
            map_cloud.set_transform(I)
        else:
            source_cloud.set_transform(I)
            map_cloud.set_transform(np.linalg.inv(frame_pose))
        source_cloud.set_enabled(self.states.view_frame)
        map_cloud.set_enabled(self.states.view_local_map)

        self.trajectory.append(frame_pose)
        if self.states.global_view:
            self._register_trajectory()

    def _register_trajectory(self):
        trajectory_cloud = self._ps.register_point_cloud(
            "trajectory",
            np.asarray(self.trajectory)[:, :3, -1],
            color=TRAJECTORY_COLOR,
        )
        trajectory_cloud.set_radius(0.3, relative=False)

    def _unregister_trajectory(self):
        self._ps.remove_point_cloud("trajectory")
