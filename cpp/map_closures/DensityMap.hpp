// MIT License
//
// Copyright (c) 2024 Saurabh Gupta, Tiziano Guadagnino, Benedikt Mersch,
// Ignacio Vizzo, Cyrill Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <vector>

namespace map_closures {

struct DensityMap {
    DensityMap(const int num_rows,
               const int num_cols,
               const double resolution,
               const Eigen::Vector2i &lower_bound);
    DensityMap(const DensityMap &other) = delete;
    DensityMap(DensityMap &&other) = default;
    DensityMap &operator=(DensityMap &&other) = default;
    DensityMap &operator=(const DensityMap &other) = delete;
    inline auto &operator()(const int x, const int y) { return grid.at<uint8_t>(x, y); }
    Eigen::Vector2i lower_bound;
    double resolution;
    cv::Mat grid;
};

DensityMap GenerateDensityMap(const std::vector<Eigen::Vector3d> &pointcloud_map,
                              const Eigen::Matrix4d &T_ground,
                              const float density_map_resolution,
                              const float density_threshold);
}  // namespace map_closures
