// MIT License

// Copyright (c) 2026 Saurabh Gupta

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
#include "VoxelMap.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <tuple>
#include <vector>

namespace {

inline Eigen::Vector3i ToVoxelCoordinates(const Eigen::Vector3d &point, const double voxel_size) {
    return Eigen::Vector3i(static_cast<int>(std::floor(point.x() / voxel_size)),
                           static_cast<int>(std::floor(point.y() / voxel_size)),
                           static_cast<int>(std::floor(point.z() / voxel_size)));
}

inline Eigen::Vector3d VoxelCenter(const Eigen::Vector3i &voxel, const double voxel_size) {
    return Eigen::Vector3d(static_cast<double>(voxel.x()) * voxel_size,
                           static_cast<double>(voxel.y()) * voxel_size,
                           static_cast<double>(voxel.z()) * voxel_size) +
           Eigen::Vector3d::Constant(voxel_size * 0.5);
}

static constexpr unsigned int min_points_for_covariance_computation = 10;

std::tuple<Eigen::Vector3d, Eigen::Vector3d> ComputeMeanAndNormal(
    const map_closures::VoxelBlock &coordinates) {
    const double num_points = static_cast<double>(coordinates.size());
    Eigen::Vector3d mean =
        std::reduce(coordinates.cbegin(), coordinates.cend(), Eigen::Vector3d().setZero()) /
        num_points;

    const Eigen::Matrix3d covariance =
        std::transform_reduce(coordinates.cbegin(), coordinates.cend(), Eigen::Matrix3d().setZero(),
                              std::plus<Eigen::Matrix3d>(),
                              [&mean](const Eigen::Vector3d &point) {
                                  Eigen::Vector3d centered = point - mean;
                                  Eigen::Matrix3d S = centered * centered.transpose();
                                  return S;
                              }) /
        (num_points - 1);
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
    Eigen::Vector3d normal = solver.eigenvectors().col(0);
    return {std::move(mean), std::move(normal)};
}

}  // namespace

namespace map_closures {
void VoxelBlock::emplace_back(const Eigen::Vector3d &p) {
    if (size() < max_points_per_normal_computation) {
        points.at(num_points) = p;
        ++num_points;
    }
}

VoxelMap::VoxelMap(const double voxel_size, const double max_distance)
    : voxel_size_(voxel_size),
      map_resolution2_(voxel_size * voxel_size /
                       static_cast<double>(max_points_per_normal_computation)),
      max_distance_(max_distance) {}

void VoxelMap::IntegrateFrame(const Vector3dVector &points, const Eigen::Matrix4d &pose) {
    const Eigen::Matrix3d R = pose.block<3, 3>(0, 0);
    const Eigen::Vector3d t = pose.block<3, 1>(0, 3);
    std::vector<Eigen::Vector3d> points_transformed(points.size());
    std::transform(points.cbegin(), points.cend(), points_transformed.begin(),
                   [&](const auto &point) { return R * point + t; });
    AddPoints(points_transformed);
}

void VoxelMap::AddPoints(const Vector3dVector &points) {
    for (const auto &point : points) {
        const Voxel voxel = ToVoxelCoordinates(point, voxel_size_);
        const auto [it, inserted] = map_.try_emplace(voxel, VoxelBlock());
        if (!inserted) {
            const VoxelBlock &voxel_block = it->second;
            if (voxel_block.size() == max_points_per_normal_computation ||
                std::any_of(voxel_block.cbegin(), voxel_block.cend(),
                            [&](const Eigen::Vector3d &voxel_point) {
                                return (voxel_point - point).squaredNorm() < map_resolution2_;
                            })) {
                continue;
            }
        }
        it->second.emplace_back(point);
    }
}

Vector3dVector VoxelMap::Pointcloud() const {
    Vector3dVector points;
    points.reserve(map_.size() * max_points_per_normal_computation);
    for (const auto &[voxel, voxel_block] : map_) {
        for (auto it = voxel_block.cbegin(); it != voxel_block.cend(); ++it) {
            points.emplace_back(*it);
        }
    }
    return points;
}

std::tuple<Vector3dVector, Vector3dVector> VoxelMap::PerVoxelMeanAndNormal() const {
    Vector3dVector voxel_means;
    voxel_means.reserve(map_.size());
    Vector3dVector voxel_normals;
    voxel_normals.reserve(map_.size());
    for (const auto &[_, voxel_block] : map_) {
        if (voxel_block.size() >= min_points_for_covariance_computation) {
            const auto &[mean, normal] = ComputeMeanAndNormal(voxel_block);
            voxel_means.emplace_back(mean);
            voxel_normals.emplace_back(normal);
        }
    }
    return {std::move(voxel_means), std::move(voxel_normals)};
}

void VoxelMap::RemovePointsFarFromLocation(const Eigen::Vector3d &origin) {
    const auto max_distance2 = max_distance_ * max_distance_;
    for (auto it = map_.begin(); it != map_.end();) {
        it = (VoxelCenter(it->first, voxel_size_) - origin).squaredNorm() >= max_distance2
                 ? map_.erase(it)
                 : std::next(it);
    }
}
}  // namespace map_closures
