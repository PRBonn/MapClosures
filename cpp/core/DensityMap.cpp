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
#include <tuple>
#include <unordered_map>
#include <vector>

using Pixel = Eigen::Vector2i;
using Point3D = Eigen::Vector3d;

namespace map_closures {
std::unordered_map<Pixel, double, PixelHash> GenerateDensityMap(
    const std::vector<Point3D> &pointcloud_map,
    const float voxel_size,
    const float density_map_threshold) {
    std::unordered_map<Pixel, double, PixelHash> density_map;
    for (const Point3D &point : pointcloud_map) {
        auto x_coord = static_cast<int>(std::floor(point[0] / voxel_size));
        auto y_coord = static_cast<int>(std::floor(point[1] / voxel_size));
        Pixel pixel(x_coord, y_coord);
        density_map[pixel] += 1.0;
    }

    // Minmax Normalization
    const auto [minval_elem, maxval_elem] = std::minmax_element(
        density_map.cbegin(), density_map.cend(),
        [](const std::pair<Pixel, double> &a, const std::pair<Pixel, double> &b) -> bool {
            return a.second < b.second;
        });
    auto min_count = minval_elem->second;
    auto max_count = maxval_elem->second;
    auto range = max_count - min_count;

    for (auto &element : density_map) {
        auto density_val = (element.second - min_count) * 255 / range;
        density_map[element.first] = density_val > density_map_threshold ? density_val : 0.0;
    }
    return density_map;
}

std::tuple<cv::Mat, Pixel> DensityMapAsImage(
    const std::unordered_map<Eigen::Vector2i, double, PixelHash> &density_map) {
    const auto [min_x_elem, max_x_elem] = std::minmax_element(
        density_map.cbegin(), density_map.cend(),
        [](const std::pair<Pixel, double> &a, const std::pair<Pixel, double> &b) -> bool {
            return a.first.x() < b.first.x();
        });

    const auto [min_y_elem, max_y_elem] = std::minmax_element(
        density_map.cbegin(), density_map.cend(),
        [](const std::pair<Pixel, double> &a, const std::pair<Pixel, double> &b) -> bool {
            return a.first.y() < b.first.y();
        });

    auto min_x = min_x_elem->first.x();
    auto max_x = max_x_elem->first.x();
    auto min_y = min_y_elem->first.y();
    auto max_y = max_y_elem->first.y();

    auto n_rows = max_x - min_x + 1;
    auto n_cols = max_y - min_y + 1;
    cv::Mat density_map_as_img(n_rows, n_cols, CV_8UC1, 0.0);

    for (const auto &element : density_map) {
        auto row = element.first.x() - min_x;
        auto col = element.first.y() - min_y;
        density_map_as_img.at<uint8_t>(row, col) = static_cast<uint8_t>(element.second);
    }

    return {density_map_as_img, Pixel(min_x, min_y)};
}
}  // namespace map_closures
