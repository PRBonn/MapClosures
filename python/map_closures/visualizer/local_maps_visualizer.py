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
import numpy as np
from matplotlib import pyplot as plt

# Button names
PREV_LOCAL_MAP = "Prev Local Map [P]"
NEXT_LOCAL_MAP = "Next Local Map [N]"
OPEN_DENSITY_VIEW = "Open Density Map Viewer [D]"
QUIT_DENSITY_VIEW = "Quit Density Map Viewer [D]"

LOCAL_MAP_COLOR = [0.8470, 0.1058, 0.3764]
PTS_SIZE = 0.06
I = np.eye(4)


class LocalMapVisualizer:
    def __init__(self, ps, gui, localmap_data):
        self._ps = ps
        self._gui = gui

        # Initialize GUI States
        self.density_viewer = None
        self.fig = None

        self.view_local_map = True
        self.local_map_id: int = 0
        self.toggle_view: bool = False
        self.global_view: bool = False
        self.view_density_map: bool = False
        self.local_map_points_size: float = PTS_SIZE

        # Create data
        self.localmap_data = localmap_data

    def matplotlib_eventloop(self):
        plt.gcf().canvas.draw()
        plt.gcf().canvas.start_event_loop(1e-6)

    def close_density_map_fig(self):
        plt.close(self.fig)

    def _pts_size_callback(self):
        changed, self.local_map_points_size = self._gui.SliderFloat(
            "##pts size", self.local_map_points_size, v_min=0.01, v_max=0.6
        )
        if changed:
            self._ps.get_point_cloud("local_map").set_radius(
                self.local_map_points_size, relative=False
            )

    def _navigate_callback(self):
        changed, self.local_map_id = self._gui.SliderInt(
            f"###Local Map Index",
            self.local_map_id,
            v_min=0,
            v_max=self.localmap_data.size - 1,
            format="Local Map ID: %d",
        )
        if changed:
            self._update_callback()

    def _previous_next_localmap_callback(self):
        if self._gui.Button(PREV_LOCAL_MAP) or self._gui.IsKeyPressed(self._gui.ImGuiKey_P):
            self.local_map_id = (self.local_map_id - 1) % self.localmap_data.size
        self._gui.SameLine()
        if self._gui.Button(NEXT_LOCAL_MAP) or self._gui.IsKeyPressed(self._gui.ImGuiKey_N):
            self.local_map_id = (self.local_map_id + 1) % self.localmap_data.size
        self._update_callback()

    def _update_callback(self):
        local_map = self._ps.register_point_cloud(
            "local_map",
            self.localmap_data.local_maps[self.local_map_id],
            color=LOCAL_MAP_COLOR,
            point_render_mode="quad",
        )
        if self.global_view:
            local_map_pose = self.localmap_data.local_map_poses[self.local_map_id]
            local_map.set_transform(local_map_pose)
        else:
            local_map.set_transform(I)
        local_map.set_enabled(self.view_local_map)
        local_map.set_radius(self.local_map_points_size, relative=False)
        if self.view_density_map:
            self.density_viewer.set_data(self.localmap_data.density_maps[self.local_map_id])
            self.matplotlib_eventloop()

    def _density_map_callback(self):
        BUTTON = OPEN_DENSITY_VIEW if not self.view_density_map else QUIT_DENSITY_VIEW
        if self._gui.Button(BUTTON) or self._gui.IsKeyPressed(self._gui.ImGuiKey_D):
            self.view_density_map = not self.view_density_map
            if self.view_density_map:
                plt.ion()
                self.fig = plt.figure()
                plt.show(block=False)
                ax = self.fig.add_subplot(1, 1, 1)
                ax.set_title("Density Map")
                self.density_viewer = ax.imshow(
                    self.localmap_data.density_maps[self.local_map_id], cmap="gray"
                )
                self.matplotlib_eventloop()
            else:
                plt.close(self.fig)
