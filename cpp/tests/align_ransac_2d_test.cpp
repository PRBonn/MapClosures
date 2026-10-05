// SPDX-License-Identifier: MIT

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "map_closures/AlignRansac2D.hpp"

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Expected a test case name\n";
        return 1;
    }
    const std::string name = argv[1];
    double degrees = 0.0;
    bool reflection = false;
    bool noncollinear = false;
    Eigen::Vector2d center(5.0, 7.0);
    if (name == "identity") {
        degrees = 0.0;
    } else if (name == "rotation_30") {
        degrees = 30.0;
    } else if (name == "rotation_90") {
        degrees = 90.0;
    } else if (name == "rotation_180") {
        degrees = 180.0;
    } else if (name == "rotation_225") {
        degrees = 225.0;
    } else if (name == "rotation_270") {
        degrees = 270.0;
    } else if (name == "noncollinear") {
        degrees = 180.0;
        noncollinear = true;
    } else if (name == "reflection_origin" || name == "reflection_offset") {
        reflection = true;
        if (name == "reflection_origin") center.setZero();
    } else {
        std::cerr << "Unknown test case: " << name << '\n';
        return 1;
    }

    Eigen::Isometry2d expected = Eigen::Isometry2d::Identity();
    expected.linear() = Eigen::Rotation2Dd(degrees * std::acos(-1.0) / 180.0).toRotationMatrix();
    expected.translation() << 2.0, -3.0;
    std::vector<Eigen::Vector2d> offsets{{-1.0, 0.0}, {1.0, 0.0}};
    if (reflection || noncollinear) {
        offsets = {{-1.0, -0.25}, {-1.0, 0.25}, {1.0, -0.25}, {1.0, 0.25}};
    }
    std::vector<map_closures::PointPair> pairs;
    Eigen::Vector2d query_mean = Eigen::Vector2d::Zero();
    for (const auto &offset : offsets) {
        const Eigen::Vector2d ref = center + offset;
        // Reflect a narrow rectangle: its best proper rigid fit is the identity
        // rotation, with residuals safely below the RANSAC inlier threshold.
        const Eigen::Vector2d query =
            reflection ? Eigen::Vector2d(ref.x(), center.y() - offset.y()) + expected.translation()
                       : expected * ref;
        pairs.emplace_back(ref, query);
        query_mean += query;
    }
    query_mean /= static_cast<double>(pairs.size());

    const auto [actual, inliers] = map_closures::RansacAlignment2D(pairs);
    const auto rotation = actual.linear();
    const double error = (actual.matrix() - expected.matrix()).norm();
    if (!actual.matrix().allFinite() || inliers != pairs.size() ||
        std::abs(rotation.determinant() - 1.0) > 1e-10 ||
        (rotation.transpose() * rotation - Eigen::Matrix2d::Identity()).norm() > 1e-10 ||
        (actual * center - query_mean).norm() > 1e-10 || error > 1e-10) {
        std::cerr << name << ": inliers=" << inliers << ", determinant=" << rotation.determinant()
                  << ", transform error=" << error << '\n';
        return 1;
    }
    return 0;
}
