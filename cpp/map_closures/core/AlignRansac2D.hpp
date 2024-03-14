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
#include <utility>
#include <vector>

namespace map_closures {
struct PointPair {
    PointPair() = default;
    PointPair(const Eigen::Vector2d &ref, const Eigen::Vector2d &query) : ref(ref), query(query) {}
    PointPair &operator+=(const PointPair &rhs) {
        this->ref += rhs.ref;
        this->query += rhs.query;
        return *this;
    }
    friend PointPair operator+(PointPair lhs, const PointPair &rhs) { return lhs += rhs; }
    friend PointPair operator/(PointPair lhs, const double divisor) {
        lhs.ref /= divisor;
        lhs.query /= divisor;
        return lhs;
    }

    Eigen::Vector2d ref = Eigen::Vector2d::Zero();
    Eigen::Vector2d query = Eigen::Vector2d::Zero();
};

struct SE2 {
    SE2() = default;
    SE2(const Eigen::Matrix2d &R, const Eigen::Vector2d &t) : R(R), t(t) {}

    friend Eigen::Vector2d operator*(const SE2 &T, const Eigen::Vector2d &p) {
        return T.R * p + T.t;
    }

    Eigen::Matrix2d R = Eigen::Matrix2d::Identity();
    Eigen::Vector2d t = Eigen::Vector2d::Zero();
};

std::pair<SE2, int> RansacAlignment2D(const std::vector<PointPair> &keypoint_pairs);
}  // namespace map_closures
