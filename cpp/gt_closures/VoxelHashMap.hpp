// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
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
//
// NOTE: This implementation is heavily inspired in the original CT-ICP VoxelHashMap implementation,
// although it was heavily modifed and drastically simplified, but if you are using this module you
// should at least acknoowledge the work from CT-ICP by giving a star on GitHub
#pragma once

#include <tsl/robin_map.h>

#include <Eigen/Core>
#include <vector>

using Voxel = Eigen::Vector3i;

template <>
struct std::hash<Voxel> {
    std::size_t operator()(const Voxel &voxel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
        return (vec[0] * 73856093 ^ vec[1] * 19349669 ^ vec[2] * 83492791);
    }
};

struct VoxelHashMap {
    VoxelHashMap(const double voxel_size, const unsigned int max_points_per_voxel = 1)
        : voxel_size_(voxel_size), max_points_per_voxel_(max_points_per_voxel) {}

    void AddPoints(const std::vector<Eigen::Vector3d> &points);
    double ComputeOverlap(const VoxelHashMap &other_map);
    int size() const { return map_.size(); }
    void clear() { map_.clear(); }

    double voxel_size_;
    unsigned int max_points_per_voxel_;
    tsl::robin_map<Voxel, std::vector<Eigen::Vector3d>> map_;
};
