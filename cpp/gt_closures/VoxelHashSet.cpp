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
#include "VoxelHashSet.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <vector>

inline Voxel PointToVoxel(const Eigen::Vector3d &point, const double voxel_size) {
    return Voxel(static_cast<int>(std::floor(point.x() / voxel_size)),
                 static_cast<int>(std::floor(point.y() / voxel_size)),
                 static_cast<int>(std::floor(point.z() / voxel_size)));
}

void VoxelHashSet::AddVoxels(const std::vector<Eigen::Vector3d> &points) {
    std::for_each(points.cbegin(), points.cend(), [&](const auto &point) {
        const auto voxel = PointToVoxel(point, voxel_size_);
        if (set_.find(voxel) == set_.end()) {
            set_.insert(voxel);
        }
    });
}

void VoxelHashSet::AddVoxels(const VoxelHashSet &other_set) {
    std::for_each(other_set.set_.cbegin(), other_set.set_.cend(), [&](const auto &voxel) {
        if (set_.find(voxel) == set_.end()) {
            set_.insert(voxel);
        }
    });
}

double VoxelHashSet::ComputeOverlap(const VoxelHashSet &other_set) {
    int overlapping_voxels = 0;
    std::for_each(other_set.set_.cbegin(), other_set.set_.cend(), [&](const auto &voxel) {
        if (set_.find(voxel) != set_.end()) {
            overlapping_voxels++;
        }
    });
    double overlap_score = static_cast<double>(overlapping_voxels) /
                           static_cast<double>(std::min(this->size(), other_set.size()));
    return overlap_score;
}
