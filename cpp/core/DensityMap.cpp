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
#include <limits>
#include <opencv2/core.hpp>
#include <utility>
#include <vector>

namespace {
struct ComparePixels {
    bool operator()(const Eigen::Vector2i &lhs, const Eigen::Vector2i &rhs) const {
        return lhs.x() < rhs.x() || (lhs.x() == rhs.x() && lhs.y() < rhs.y());
    }
};
using PointCounterType = std::map<Eigen::Vector2i, double, ComparePixels>;
}  // namespace

namespace map_closures {
std::pair<cv::Mat, Eigen::Vector2i> GenerateDensityMap(const std::vector<Eigen::Vector3d> &pcd,
                                                       const float density_map_resolution,
                                                       const float density_threshold) {
    PointCounterType point_counter;
    double max_points = std::numeric_limits<double>::min();
    double min_points = std::numeric_limits<double>::max();
    std::for_each(pcd.cbegin(), pcd.cend(), [&](const Eigen::Vector3d &point) {
        const Eigen::Vector2i pixel = (point.head<2>() / density_map_resolution).cast<int>();
        auto &num_points = point_counter[pixel];
        point_counter[pixel] += 1.0;
        max_points = std::max(max_points, num_points);
        min_points = std::min(min_points, num_points);
    });
    const Eigen::Vector2i &lower_bound_coordinates = point_counter.cbegin()->first;
    const Eigen::Vector2i &upper_bound_coordinates = point_counter.crbegin()->first;
    const auto rows_and_columns = upper_bound_coordinates - lower_bound_coordinates;
    const auto n_rows = rows_and_columns.x() + 1;
    const auto n_cols = rows_and_columns.y() + 1;
    const double min_max_normalizer = max_points - min_points;
    cv::Mat density_img(n_rows, n_cols, CV_8UC1, 0.0);
    std::for_each(point_counter.cbegin(), point_counter.cend(), [&](const auto &element) {
        const auto &[pixel, point_counter] = element;
        auto density = (point_counter - min_points) * 255 / min_max_normalizer;
        density = density > density_threshold ? density : 0.0;
        const auto px = pixel - lower_bound_coordinates;
        density_img.at<uint8_t>(px.x(), px.y()) = static_cast<uint8_t>(density);
    });

    return {density_img, lower_bound_coordinates};
}
}  // namespace map_closures
