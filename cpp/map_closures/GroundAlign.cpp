// MIT License
//
// Copyright (c) 2025 Saurabh Gupta, Tiziano Guadagnino, Benedikt Mersch,
// Niklas Trekel, Meher Malladi, and Cyrill Stachniss.
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

#include "GroundAlign.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <numeric>
#include <sophus/se3.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
struct PixelHash {
    size_t operator()(const Eigen::Vector2i &pixel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(pixel.data());
        return (vec[0] * 73856093 ^ vec[1] * 19349669);
    }
};

void TransformPoints(const Sophus::SE3d &T, Vector3dVector &pointcloud) {
    std::transform(pointcloud.cbegin(), pointcloud.cend(), pointcloud.begin(),
                   [&](const auto &point) { return T * point; });
}

struct VoxelMeanAndNormal {
    Eigen::Vector3d mean;
    Eigen::Vector3d normal;
};

auto PointToPixel = [](const Eigen::Vector3d &pt) -> Eigen::Vector2i {
    return Eigen::Vector2i(static_cast<int>(std::floor(pt.x())),
                           static_cast<int>(std::floor(pt.y())));
};

std::pair<Vector3dVector, Sophus::SE3d> SampleGroundPoints(const Vector3dVector &voxel_means,
                                                           const Vector3dVector &voxel_normals) {
    std::unordered_map<Eigen::Vector2i, VoxelMeanAndNormal, PixelHash> lowest_voxel_hash_map;

    for (size_t index = 0; index < voxel_means.size(); ++index) {
        const Eigen::Vector3d &mean = voxel_means[index];
        const Eigen::Vector3d &normal = voxel_normals[index];
        const Eigen::Vector2i pixel = PointToPixel(mean);

        auto it = lowest_voxel_hash_map.find(pixel);
        if (it == lowest_voxel_hash_map.end()) {
            lowest_voxel_hash_map.emplace(pixel, VoxelMeanAndNormal{mean, normal});
        } else if (mean.z() < it->second.mean.z()) {
            it->second = VoxelMeanAndNormal{mean, normal};
        }
    }

    std::vector<VoxelMeanAndNormal> low_lying_voxels(lowest_voxel_hash_map.size());
    std::transform(lowest_voxel_hash_map.cbegin(), lowest_voxel_hash_map.cend(),
                   low_lying_voxels.begin(), [](const auto &entry) { return entry.second; });

    const Eigen::Matrix3d covariance_matrix =
        std::transform_reduce(
            low_lying_voxels.cbegin(), low_lying_voxels.cend(), Eigen::Matrix3d().setZero(),
            std::plus<Eigen::Matrix3d>(),
            [&](const auto &voxel) { return voxel.normal * voxel.normal.transpose(); }) /
        static_cast<double>(low_lying_voxels.size() - 1);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(covariance_matrix);
    Eigen::Vector3d largest_eigenvector = eigensolver.eigenvectors().col(2);
    largest_eigenvector =
        (largest_eigenvector.z() < 0) ? -1.0 * largest_eigenvector : largest_eigenvector;
    const Eigen::Vector3d axis = largest_eigenvector.cross(Eigen::Vector3d(0.0, 0.0, 1.0));
    const double angle = std::acos(std::clamp(largest_eigenvector.z(), -1.0, 1.0));
    const double axis_norm = axis.norm();

    const Eigen::Matrix3d R = axis_norm > 1e-3
                                  ? Eigen::AngleAxisd(angle, axis / axis_norm).toRotationMatrix()
                                  : Eigen::Matrix3d::Identity();

    Eigen::Vector3d ground_centroid(0.0, 0.0, 0.0);
    Vector3dVector ground_samples;
    ground_samples.reserve(low_lying_voxels.size());
    std::for_each(low_lying_voxels.cbegin(), low_lying_voxels.cend(), [&](const auto &voxel) {
        if (std::abs(voxel.normal.dot(largest_eigenvector)) > 0.95) {
            ground_centroid += voxel.mean;
            ground_samples.emplace_back(voxel.mean);
        }
    });
    ground_samples.shrink_to_fit();
    ground_centroid /= static_cast<double>(ground_samples.size());

    const double z_shift = R.row(2) * ground_centroid;
    const Sophus::SE3d T_init(R, Eigen::Vector3d(0.0, 0.0, -1.0 * z_shift));
    return std::make_pair(ground_samples, T_init);
}

using LinearSystem = std::pair<Eigen::Matrix3d, Eigen::Vector3d>;
LinearSystem BuildLinearSystem(const Vector3dVector &points) {
    auto compute_jacobian_and_residual = [](const auto &point) {
        const double residual = point.z();
        Eigen::Matrix<double, 1, 3> J;
        J(0, 0) = 1.0;
        J(0, 1) = point.y();
        J(0, 2) = -point.x();
        return std::make_pair(J, residual);
    };

    auto sum_linear_systems = [](LinearSystem a, const LinearSystem &b) {
        a.first += b.first;
        a.second += b.second;
        return a;
    };

    const auto &[H, b] =
        std::transform_reduce(points.cbegin(), points.cend(),
                              LinearSystem(Eigen::Matrix3d::Zero(), Eigen::Vector3d::Zero()),
                              sum_linear_systems, [&](const auto &point) {
                                  const auto &[J, residual] = compute_jacobian_and_residual(point);
                                  const double w = std::exp(-1.0 * residual * residual);
                                  return LinearSystem(J.transpose() * w * J,          // JTJ
                                                      J.transpose() * w * residual);  // JTr
                              });
    return {H, b};
}

static constexpr double convergence_threshold = 1e-3;
static constexpr int max_iterations = 10;
}  // namespace

namespace map_closures {
Eigen::Matrix4d AlignToLocalGround(const Vector3dVector &voxel_means,
                                   const Vector3dVector &voxel_normals) {
    auto [ground_samples, T] = SampleGroundPoints(voxel_means, voxel_normals);
    TransformPoints(T, ground_samples);
    for (int iters = 0; iters < max_iterations; iters++) {
        const auto &[H, b] = BuildLinearSystem(ground_samples);
        const Eigen::Vector3d &dx = H.ldlt().solve(-b);
        Eigen::Matrix<double, 6, 1> se3 = Eigen::Matrix<double, 6, 1>::Zero();
        se3.block<3, 1>(2, 0) = dx;
        Sophus::SE3d estimation(Sophus::SE3d::exp(se3));
        TransformPoints(estimation, ground_samples);
        T = estimation * T;
        if (dx.norm() < convergence_threshold) break;
    }
    return T.matrix();
}
}  // namespace map_closures
