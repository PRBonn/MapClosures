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
#include <tuple>
#include <unordered_map>
#include <vector>

namespace map_closures {
struct PixelHash {
    size_t operator()(const Eigen::Vector2i &pixel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(pixel.data());
        return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349669);
    }
};

std::unordered_map<Eigen::Vector2i, double, PixelHash> GenerateDensityMap(
    const std::vector<Eigen::Vector3d> &pointcloud_map,
    const float voxel_size,
    const float density_map_threshold);

std::tuple<cv::Mat, Eigen::Vector2i> DensityMapAsImage(
    const std::unordered_map<Eigen::Vector2i, double, PixelHash> &density_map);
}  // namespace map_closures
