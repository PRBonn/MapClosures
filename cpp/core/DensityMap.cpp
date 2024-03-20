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
    bool operator()(const Eigen::Array2i &lhs, const Eigen::Array2i &rhs) const {
        return lhs.x() < rhs.x() || (lhs.x() == rhs.x() && lhs.y() < rhs.y());
    }
};
using PointCounterType = std::map<Eigen::Array2i, double, ComparePixels>;
constexpr int max_int = std::numeric_limits<int>::max();
constexpr int min_int = std::numeric_limits<int>::min();
}  // namespace

namespace map_closures {

DensityMap::DensityMap(const int num_rows, const int num_cols, const double resolution_)
    : lower_bound(0, 0), resolution(resolution_), grid(num_rows, num_cols, CV_8UC1, 0.0) {}

DensityMap GenerateDensityMap(const std::vector<Eigen::Vector3d> &pcd,
                              const float density_map_resolution,
                              const float density_threshold) {
    PointCounterType point_counter;

    double max_points = std::numeric_limits<double>::min();
    double min_points = std::numeric_limits<double>::max();
    Eigen::Array2i lower_bound_coordinates = Eigen::Array2i::Constant(max_int);
    Eigen::Array2i upper_bound_coordinates = Eigen::Array2i::Constant(min_int);

    auto Discretize2D = [&density_map_resolution](const Eigen::Vector3d &p) -> Eigen::Array2i {
        return (p.head<2>().array() / density_map_resolution).cast<int>();
    };
    std::for_each(pcd.cbegin(), pcd.cend(), [&](const Eigen::Vector3d &point) {
        const auto pixel = Discretize2D(point);
        auto &num_points = point_counter[pixel];
        point_counter[pixel] += 1.0;
        max_points = std::max(max_points, num_points);
        min_points = std::min(min_points, num_points);
        lower_bound_coordinates = lower_bound_coordinates.min(pixel);
        upper_bound_coordinates = upper_bound_coordinates.max(pixel);
    });
    const auto rows_and_columns = upper_bound_coordinates - lower_bound_coordinates;
    const auto n_rows = rows_and_columns.x() + 1;
    const auto n_cols = rows_and_columns.y() + 1;

    const double min_max_normalizer = max_points - min_points;
    DensityMap density_map(n_rows, n_cols, density_map_resolution);
    density_map.lower_bound = lower_bound_coordinates;
    std::for_each(pcd.cbegin(), pcd.cend(), [&](const auto &point) {
        const auto pixel = Discretize2D(point);
        double raw_density = (point_counter.at(pixel) - min_points) / min_max_normalizer;
        raw_density = raw_density > density_threshold ? raw_density : 0.0;
        uint8_t discretized_density = 255 * raw_density;
        const auto px = pixel - lower_bound_coordinates;
        density_map(px.x(), px.y()) = discretized_density;
    });

    return density_map;
}
}  // namespace map_closures
