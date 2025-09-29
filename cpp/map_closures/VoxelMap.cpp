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

static constexpr unsigned int min_points_for_covariance_computation = 10;

std::tuple<Eigen::Vector3d, Eigen::Vector3d> ComputeMeanAndNormal(
    const map_closures::VoxelBlock &coordinates) {
    const double num_points = static_cast<double>(coordinates.size());
    const Eigen::Vector3d &mean =
        std::reduce(coordinates.cbegin(), coordinates.cend(), Eigen::Vector3d().setZero()) /
        num_points;

    const Eigen::Matrix3d &covariance =
        std::transform_reduce(coordinates.cbegin(), coordinates.cend(), Eigen::Matrix3d().setZero(),
                              std::plus<Eigen::Matrix3d>(),
                              [&mean](const Eigen::Vector3d &point) {
                                  const Eigen::Vector3d &centered = point - mean;
                                  const Eigen::Matrix3d S = centered * centered.transpose();
                                  return S;
                              }) /
        (num_points - 1);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
    const Eigen::Vector3d normal = solver.eigenvectors().col(0);
    return std::make_tuple(mean, normal);
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
      map_resolution_(voxel_size /
                      static_cast<double>(std::sqrt(max_points_per_normal_computation))),
      max_distance_(max_distance) {}

void VoxelMap::IntegrateFrame(const Vector3dVector &points, const Eigen::Matrix4d &pose) {
    Vector3dVector points_transformed(points.size());
    const Eigen::Matrix3d &R = pose.block<3, 3>(0, 0);
    const Eigen::Vector3d &t = pose.block<3, 1>(0, 3);
    std::transform(points.cbegin(), points.cend(), points_transformed.begin(),
                   [&](const auto &point) { return R * point + t; });
    AddPoints(points_transformed);
}

void VoxelMap::AddPoints(const Vector3dVector &points) {
    std::for_each(points.cbegin(), points.cend(), [&](const Eigen::Vector3d &point) {
        const Voxel voxel = ToVoxelCoordinates(point, voxel_size_);
        const auto &[it, inserted] = map_.insert({voxel, VoxelBlock()});
        if (!inserted) {
            const VoxelBlock &voxel_block = it->second;
            if (voxel_block.size() == max_points_per_normal_computation ||
                std::any_of(voxel_block.cbegin(), voxel_block.cend(),
                            [&](const Eigen::Vector3d &voxel_point) {
                                return (voxel_point - point).norm() < map_resolution_;
                            })) {
                return;
            }
        }
        it->second.emplace_back(point);
    });
}

Vector3dVector VoxelMap::Pointcloud() const {
    Vector3dVector points;
    points.reserve(map_.size() * max_points_per_normal_computation);
    std::for_each(map_.cbegin(), map_.cend(), [&](const auto &map_element) {
        const VoxelBlock &voxel_block = map_element.second;
        std::for_each(voxel_block.cbegin(), voxel_block.cend(), [&](const Eigen::Vector3d &p) {
            points.emplace_back(p.template cast<double>());
        });
    });
    points.shrink_to_fit();
    return points;
}

std::tuple<Vector3dVector, Vector3dVector> VoxelMap::PerVoxelMeanAndNormal() const {
    Vector3dVector voxel_means;
    voxel_means.reserve(map_.size());
    Vector3dVector voxel_normals;
    voxel_normals.reserve(map_.size());
    std::for_each(map_.cbegin(), map_.cend(), [&](const auto &map_element) {
        const VoxelBlock &voxel_block = map_element.second;
        if (voxel_block.size() >= min_points_for_covariance_computation) {
            const auto &[mean, normal] = ComputeMeanAndNormal(voxel_block);
            voxel_means.emplace_back(mean);
            voxel_normals.emplace_back(normal);
        }
    });
    voxel_means.shrink_to_fit();
    voxel_normals.shrink_to_fit();
    return std::make_tuple(voxel_means, voxel_normals);
}

void VoxelMap::RemovePointsFarFromLocation(const Eigen::Vector3d &origin) {
    const double max_distance2 = max_distance_ * max_distance_;
    for (auto it = map_.begin(); it != map_.end();) {
        const auto &[voxel, voxel_points] = *it;
        const Eigen::Vector3d &pt = voxel_points.front();
        if ((pt - origin).squaredNorm() >= (max_distance2)) {
            it = map_.erase(it);
        } else {
            ++it;
        }
    }
}
}  // namespace map_closures
