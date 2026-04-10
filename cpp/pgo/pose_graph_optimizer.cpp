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
#include "pose_graph_optimizer.hpp"

#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/stuff/macros.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/vertex_se3.h>

#include <algorithm>
#include <memory>
#include <utility>

namespace {
static constexpr double epsilon = 1e-6;
}
// clang-format off
namespace g2o {
G2O_REGISTER_TYPE(VERTEX_SE3:QUAT, VertexSE3)
G2O_REGISTER_TYPE(EDGE_SE3:QUAT, EdgeSE3)
}  // namespace g2o
// clang-format on

namespace pgo {
using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>>;
using LinearSolverType = g2o::LinearSolverCholmod<BlockSolverType::PoseMatrixType>;
using AlgorithmType = g2o::OptimizationAlgorithmDogleg;

PoseGraphOptimizer::PoseGraphOptimizer(const int max_iterations) : max_iterations_(max_iterations) {
    graph = std::make_unique<g2o::SparseOptimizer>();
    graph->setVerbose(true);

    auto solver =
        new AlgorithmType(std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));

    auto terminateAction = new g2o::SparseOptimizerTerminateAction;
    terminateAction->setGainThreshold(epsilon);
    graph->addPostIterationAction(terminateAction);
    graph->setAlgorithm(solver);
}

void PoseGraphOptimizer::fixVariable(const int id) { graph->vertex(id)->setFixed(true); }

void PoseGraphOptimizer::addVariable(const int id, const Eigen::Matrix4d &T) {
    Eigen::Isometry3d pose(T);
    g2o::VertexSE3 *variable = new g2o::VertexSE3();
    variable->setId(id);
    variable->setEstimate(pose);
    graph->addVertex(variable);
}

void PoseGraphOptimizer::addFactor(const int id_source,
                                   const int id_target,
                                   const Eigen::Matrix4d &T,
                                   const Eigen::Matrix6d &information_matrix,
                                   const bool robust_kernel) {
    Eigen::Isometry3d relative_pose(T);
    g2o::EdgeSE3 *factor = new g2o::EdgeSE3();
    factor->setVertex(0, graph->vertex(id_target));
    factor->setVertex(1, graph->vertex(id_source));
    factor->setInformation(information_matrix);
    factor->setMeasurement(relative_pose);
    if (robust_kernel) {
        g2o::RobustKernelDCS *rk = new g2o::RobustKernelDCS;
        rk->setDelta(1.0);
        factor->setRobustKernel(rk);
    }
    graph->addEdge(factor);
}

PoseGraphOptimizer::PoseIDMap PoseGraphOptimizer::estimates() const {
    const g2o::HyperGraph::VertexIDMap &variables = graph->vertices();
    PoseIDMap poses;
    std::transform(variables.cbegin(), variables.cend(), std::inserter(poses, poses.end()),
                   [](const auto &id_var) {
                       const auto &[id, v] = id_var;
                       Eigen::Isometry3d pose = static_cast<g2o::VertexSE3 *>(v)->estimate();
                       return std::make_pair(id, pose.matrix());
                   });
    return poses;
}

void PoseGraphOptimizer::optimize() {
    graph->initializeOptimization();
    graph->optimize(max_iterations_);
}
}  // namespace pgo
