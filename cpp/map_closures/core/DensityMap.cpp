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

#include "DensityMap.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <opencv2/core.hpp>
#include <utility>
#include <vector>

namespace {
struct ComparePixels {
    bool operator()(const Eigen::Vector2i &lhs, const Eigen::Vector2i &rhs) const {
        return lhs.x() < rhs.x() || (lhs.x() == rhs.x() && lhs.y() < rhs.y());
    }
};
using DensityMapType = std::map<Eigen::Vector2i, double, ComparePixels>;
}  // namespace

namespace map_closures {
std::pair<cv::Mat, Eigen::Vector2i> GenerateDensityMap(
    const std::vector<Eigen::Vector3d> &pointcloud_map,
    const float density_map_resolution,
    const float density_threshold) {
    DensityMapType density_map;
    int min_x = std::numeric_limits<int>::max();
    int min_y = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::min();
    int max_y = std::numeric_limits<int>::min();
    double max_points = std::numeric_limits<double>::min();
    double min_points = std::numeric_limits<double>::max();

    std::for_each(
        pointcloud_map.cbegin(), pointcloud_map.cend(), [&](const Eigen::Vector3d &point) {
            auto x_coord = static_cast<int>(std::floor(point[0] / density_map_resolution));
            auto y_coord = static_cast<int>(std::floor(point[1] / density_map_resolution));
            Eigen::Vector2i pixel(x_coord, y_coord);
            density_map[pixel] += 1.0;
            auto pixel_density = density_map[pixel];
            if (pixel_density > max_points) {
                max_points = pixel_density;
            } else if (pixel_density < min_points) {
                min_points = pixel_density;
            }
            if (x_coord < min_x) {
                min_x = x_coord;
            } else if (x_coord > max_x) {
                max_x = x_coord;
            }

            if (y_coord < min_y) {
                min_y = y_coord;
            } else if (y_coord > max_y) {
                max_y = y_coord;
            }
        });
    auto lower_bound_coordinates = Eigen::Vector2i(min_x, min_y);

    auto n_rows = max_x - min_x + 1;
    auto n_cols = max_y - min_y + 1;
    const double range = max_points - min_points;
    cv::Mat density_img(n_rows, n_cols, CV_8UC1, 0.0);
    std::for_each(density_map.cbegin(), density_map.cend(), [&](const auto &pixel) {
        auto density_val = (pixel.second - min_points) * 255 / range;
        density_val = density_val > density_threshold ? density_val : 0.0;
        auto row_num = pixel.first.x() - min_x;
        auto col_num = pixel.first.y() - min_y;
        density_img.at<uint8_t>(row_num, col_num) = static_cast<uint8_t>(density_val);
    });

    return {density_img, lower_bound_coordinates};
}
}  // namespace map_closures
