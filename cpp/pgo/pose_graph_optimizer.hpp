// MIT License

// Copyright (c) 2025 Tiziano Guadagnino

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
#include <g2o/core/sparse_optimizer.h>

#include <Eigen/Geometry>
#include <fstream>
#include <memory>
#include <string>

namespace Eigen {
using Matrix4d = Eigen::Matrix<double, 4, 4>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
}  // namespace Eigen
namespace pgo {
class PoseGraphOptimizer {
public:
    using PoseIDMap = std::map<int, Eigen::Matrix4d>;
    explicit PoseGraphOptimizer(const int max_iterations);

    void fixVariable(const int id);
    void addVariable(const int id, const Eigen::Matrix4d &T);

    void addFactor(const int id_source,
                   const int id_target,
                   const Eigen::Matrix4d &T,
                   const Eigen::Matrix6d &information_matrix,
                   const bool robust_kernel);

    [[nodiscard]] PoseIDMap estimates() const;

    void optimize();

private:
    std::unique_ptr<g2o::SparseOptimizer> graph;
    int max_iterations_;
};
}  // namespace pgo
