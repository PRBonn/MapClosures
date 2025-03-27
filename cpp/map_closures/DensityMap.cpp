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
#include <vector>

namespace {
struct ComparePixels {
    bool operator()(const Eigen::Array2i &lhs, const Eigen::Array2i &rhs) const {
        return lhs.x() < rhs.x() || (lhs.x() == rhs.x() && lhs.y() < rhs.y());
    }
};
constexpr int max_int = std::numeric_limits<int>::max();
constexpr int min_int = std::numeric_limits<int>::min();
}  // namespace

namespace map_closures {

DensityMap::DensityMap(const int num_rows,
                       const int num_cols,
                       const double resolution,
                       const Eigen::Vector2i &lower_bound)
    : lower_bound(lower_bound), resolution(resolution), grid(num_rows, num_cols, CV_8UC1, 0.0) {}

DensityMap GenerateDensityMap(const std::vector<Eigen::Vector3d> &pcd,
                              const Eigen::Matrix4d &T_ground,
                              const float density_map_resolution,
                              const float density_threshold) {
    double max_points = std::numeric_limits<double>::min();
    double min_points = std::numeric_limits<double>::max();
    Eigen::Array2i lower_bound_coordinates = Eigen::Array2i::Constant(max_int);
    Eigen::Array2i upper_bound_coordinates = Eigen::Array2i::Constant(min_int);

    auto Discretize2D = [&](const Eigen::Vector3d &p) -> Eigen::Array2i {
        return ((T_ground.block<3, 3>(0, 0) * p + T_ground.block<3, 1>(0, 3)).head<2>() /
                density_map_resolution)
            .array()
            .floor()
            .cast<int>();
    };
    std::vector<Eigen::Array2i> pixels(pcd.size());
    std::transform(pcd.cbegin(), pcd.cend(), pixels.begin(), [&](const Eigen::Vector3d &point) {
        const auto &pixel = Discretize2D(point);
        lower_bound_coordinates = lower_bound_coordinates.min(pixel);
        upper_bound_coordinates = upper_bound_coordinates.max(pixel);
        return pixel;
    });
    const auto rows_and_columns = upper_bound_coordinates - lower_bound_coordinates;
    const auto n_rows = rows_and_columns.x() + 1;
    const auto n_cols = rows_and_columns.y() + 1;

    cv::Mat counting_grid(n_rows, n_cols, CV_64FC1, 0.0);
    std::for_each(pixels.cbegin(), pixels.cend(), [&](const auto &pixel) {
        const auto px = pixel - lower_bound_coordinates;
        counting_grid.at<double>(px.x(), px.y()) += 1;
        max_points = std::max(max_points, counting_grid.at<double>(px.x(), px.y()));
        min_points = std::min(min_points, counting_grid.at<double>(px.x(), px.y()));
    });

    DensityMap density_map(n_rows, n_cols, density_map_resolution, lower_bound_coordinates);
    counting_grid.forEach<double>([&](const double count, const int pos[]) {
        auto density = (count - min_points) / (max_points - min_points);
        density = density > density_threshold ? density : 0.0;
        density_map(pos[0], pos[1]) = static_cast<uint8_t>(255 * density);
    });

    return density_map;
}
}  // namespace map_closures
