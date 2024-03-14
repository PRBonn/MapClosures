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

#include "AlignRansac2D.hpp"

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>
#include <random>
#include <utility>
#include <vector>

namespace {
map_closures::SE2 KabschUmeyamaAlignment2D(
    const std::vector<map_closures::PointPair> &keypoint_pairs) {
    auto mean =
        std::reduce(keypoint_pairs.cbegin(), keypoint_pairs.cend(), map_closures::PointPair()) /
        keypoint_pairs.size();
    auto covariance_matrix = std::transform_reduce(
        keypoint_pairs.cbegin(), keypoint_pairs.cend(), Eigen::Matrix2d().setZero(),
        std::plus<Eigen::Matrix2d>(), [&](const auto &keypoint_pair) {
            return (keypoint_pair.ref - mean.ref) *
                   ((keypoint_pair.query - mean.query).transpose());
        });

    Eigen::JacobiSVD<Eigen::Matrix2d> svd(covariance_matrix,
                                          Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2d R = svd.matrixV() * svd.matrixU().transpose();
    R = R.determinant() > 0 ? R : -R;
    auto t = mean.query - R * mean.ref;

    return map_closures::SE2(R, t);
}

static constexpr double inliers_distance_threshold = 3.0;

// RANSAC Parameters
static constexpr double inliers_ratio = 0.3;
static constexpr double probability_success = 0.999;
static constexpr int min_points = 2;
static constexpr int __RANSAC_TRIALS__ = std::ceil(
    std::log(1.0 - probability_success) / std::log(1.0 - std::pow(inliers_ratio, min_points)));
}  // namespace

namespace map_closures {
std::pair<SE2, int> RansacAlignment2D(const std::vector<PointPair> &keypoint_pairs) {
    const size_t max_inliers = keypoint_pairs.size();

    std::vector<PointPair> sample_keypoint_pairs(2);
    std::vector<int> inlier_indices;
    inlier_indices.reserve(max_inliers);

    std::vector<int> optimal_inlier_indices;
    optimal_inlier_indices.reserve(max_inliers);

    int iter = 0;
    while (iter++ < __RANSAC_TRIALS__) {
        inlier_indices.clear();

        std::sample(keypoint_pairs.begin(), keypoint_pairs.end(), sample_keypoint_pairs.begin(), 2,
                    std::mt19937{std::random_device{}()});
        auto T = KabschUmeyamaAlignment2D(sample_keypoint_pairs);

        int index = 0;
        std::for_each(keypoint_pairs.cbegin(), keypoint_pairs.cend(),
                      [&](const auto &keypoint_pair) {
                          if ((T * keypoint_pair.ref - keypoint_pair.query).norm() <
                              inliers_distance_threshold)
                              inlier_indices.emplace_back(index);
                          index++;
                      });

        if (inlier_indices.size() > optimal_inlier_indices.size()) {
            optimal_inlier_indices = inlier_indices;
        }
    }

    const int num_inliers = optimal_inlier_indices.size();
    std::vector<PointPair> inlier_keypoint_pairs(num_inliers);
    std::transform(optimal_inlier_indices.cbegin(), optimal_inlier_indices.cend(),
                   inlier_keypoint_pairs.begin(),
                   [&](const auto index) { return keypoint_pairs[index]; });
    auto T = KabschUmeyamaAlignment2D(inlier_keypoint_pairs);
    return {T, num_inliers};
}
}  // namespace map_closures
