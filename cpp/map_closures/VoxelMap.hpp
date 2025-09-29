// MIT License

// Copyright (c) 2025 Saurabh Gupta, Tiziano Guadagnino, Benedikt Mersch,
// Niklas Trekel, Meher Malladi, and Cyrill Stachniss.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <Eigen/Core>
#include <array>
#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <vector>

using Vector3dVector = std::vector<Eigen::Vector3d>;

using Voxel = Eigen::Vector3i;
template <>
struct std::hash<Voxel> {
    std::size_t operator()(const Voxel &voxel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
        return (vec[0] * 73856093 ^ vec[1] * 19349669 ^ vec[2] * 83492791);
    }
};

// Same default as Open3d
static constexpr unsigned int max_points_per_normal_computation = 20;

namespace map_closures {

struct VoxelBlock {
    void emplace_back(const Eigen::Vector3d &point);
    inline constexpr size_t size() const { return num_points; }
    auto cbegin() const { return points.cbegin(); }
    auto cend() const { return std::next(points.cbegin(), num_points); }
    auto front() const { return points.front(); }
    std::array<Eigen::Vector3d, max_points_per_normal_computation> points;
    size_t num_points = 0;
};

struct VoxelMap {
    explicit VoxelMap(const double voxel_size, const double max_distance);

    inline void Clear() { map_.clear(); }
    inline bool Empty() const { return map_.empty(); }
    inline size_t NumVoxels() const { return map_.size(); }

    void IntegrateFrame(const Vector3dVector &points, const Eigen::Matrix4d &pose);
    void AddPoints(const Vector3dVector &points);

    Vector3dVector Pointcloud() const;
    std::tuple<Vector3dVector, Vector3dVector> PerVoxelMeanAndNormal() const;
    void RemovePointsFarFromLocation(const Eigen::Vector3d &origin);

    double voxel_size_;
    double map_resolution_;
    double max_distance_;
    std::unordered_map<Voxel, VoxelBlock> map_;
};
}  // namespace map_closures
