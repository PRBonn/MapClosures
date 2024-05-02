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
#include <tuple>
#include <vector>

using Eigen::Matrix4d, Eigen::Matrix2d;
using Eigen::Vector2d;
using KeyPoints2D = std::vector<Vector2d>;

namespace {
std::tuple<Vector2d, Vector2d, Matrix2d> computeMeanAndCovariance(const KeyPoints2D &A,
                                                                  const KeyPoints2D &B) {
    Vector2d A_bar = std::accumulate(A.cbegin(), A.cend(), Vector2d().setZero()) / A.size();
    Vector2d B_bar = std::accumulate(B.cbegin(), B.cend(), Vector2d().setZero()) / B.size();

    Matrix2d Cov = std::transform_reduce(
        A.cbegin(), A.cend(), B.cbegin(), Matrix2d().setZero(), std::plus<Matrix2d>(),
        [&](const auto &a, const auto &b) { return (a - A_bar) * ((b - B_bar).transpose()); });
    return {A_bar, B_bar, Cov};
}

KeyPoints2D applyRigidBodyTransform(const KeyPoints2D &pts, const Matrix2d &R, const Vector2d &tr) {
    KeyPoints2D pts_;
    pts_.reserve(pts.size());
    std::for_each(pts.cbegin(), pts.cend(),
                  [&](const auto &pt) { pts_.emplace_back(R * pt + tr); });
    return pts_;
}
}  // namespace

static constexpr int inliers_threshold = 3.0;
namespace map_closures {
std::tuple<Matrix2d, Vector2d, int> ICPRansac2D(const KeyPoints2D &ref_keypts,
                                                const KeyPoints2D &query_keypts,
                                                const std::vector<double> &weights) {
    std::discrete_distribution<int> rig(weights.cbegin(), weights.cend());
    std::mt19937_64 gen(std::random_device{}());
    int best_inliers_count = 0;
    Matrix2d best_Rot;
    Vector2d best_tr;

    int n_iters = 0;
    int max_iters = 0.5 * weights.size() * (weights.size() - 1);
    while (n_iters++ < max_iters) {
        int idx_1 = -1, idx_2 = -1;
        while (idx_1 == idx_2) {
            idx_1 = rig(gen);
            idx_2 = rig(gen);
        }

        auto [P_bar, Q_bar, Cov] = computeMeanAndCovariance(
            {ref_keypts[idx_1], ref_keypts[idx_2]}, {query_keypts[idx_1], query_keypts[idx_2]});

        Eigen::JacobiSVD<Matrix2d> svd(Cov, Eigen::ComputeFullU | Eigen::ComputeFullV);

        Matrix2d Rot = svd.matrixV() * svd.matrixU().transpose();
        if (Rot.determinant() < 0) {
            Rot = -Rot;
        }

        auto tr = Q_bar - Rot * P_bar;

        auto Q_tf = applyRigidBodyTransform(ref_keypts, Rot, tr);

        auto inliers_count =
            std::inner_product(query_keypts.cbegin(), query_keypts.cend(), Q_tf.cbegin(), 0.0,
                               std::plus<>(), [&](const Vector2d &a, const Vector2d &b) {
                                   return (a - b).norm() < inliers_threshold;
                               });

        if (inliers_count >= best_inliers_count) {
            best_inliers_count = inliers_count;
            best_Rot = Rot;
            best_tr = tr;
        }
    }
    return {best_Rot, best_tr, best_inliers_count};
}
}  // namespace map_closures
