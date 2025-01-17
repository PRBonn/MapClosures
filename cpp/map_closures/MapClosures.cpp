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

#include "MapClosures.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <utility>
#include <vector>

#include "AlignRansac2D.hpp"
#include "DensityMap.hpp"
#include "srrg_hbst/types/binary_tree.hpp"

namespace {
// fixed parameters for OpenCV ORB Features
static constexpr float scale_factor = 1.00;
static constexpr int n_levels = 1;
static constexpr int first_level = 0;
static constexpr int WTA_K = 2;
static constexpr int nfeatures = 500;
static constexpr int edge_threshold = 31;
static constexpr int score_type = 0;
static constexpr int patch_size = 31;
static constexpr int fast_threshold = 35;
}  // namespace

namespace map_closures {
MapClosures::MapClosures() : config_(Config()) {
    orb_extractor_ =
        cv::ORB::create(nfeatures, scale_factor, n_levels, edge_threshold, first_level, WTA_K,
                        cv::ORB::ScoreType(score_type), patch_size, fast_threshold);
}

MapClosures::MapClosures(const Config &config) : config_(config) {
    orb_extractor_ =
        cv::ORB::create(nfeatures, scale_factor, n_levels, edge_threshold, first_level, WTA_K,
                        cv::ORB::ScoreType(score_type), patch_size, fast_threshold);
}

std::vector<int> MapClosures::MatchAndAdd(const int id,
                                          const std::vector<Eigen::Vector3d> &local_map) {
    DensityMap density_map =
        GenerateDensityMap(local_map, config_.density_map_resolution, config_.density_threshold);
    cv::Mat orb_descriptors;
    std::vector<cv::KeyPoint> orb_keypoints;
    orb_extractor_->detectAndCompute(density_map.grid, cv::noArray(), orb_keypoints,
                                     orb_descriptors);

    auto matcher = cv::BFMatcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> bf_matches;
    matcher.knnMatch(orb_descriptors, orb_descriptors, bf_matches, 2);

    std::for_each(orb_keypoints.begin(), orb_keypoints.end(), [&](cv::KeyPoint &keypoint) {
        keypoint.pt.x = keypoint.pt.x + static_cast<float>(density_map.lower_bound.y());
        keypoint.pt.y = keypoint.pt.y + static_cast<float>(density_map.lower_bound.x());
    });

    std::vector<Matchable *> hbst_matchable;
    hbst_matchable.reserve(orb_descriptors.rows);
    std::for_each(bf_matches.cbegin(), bf_matches.cend(), [&](const auto &bf_match) {
        if (bf_match[1].distance > 35.0) {
            auto index_descriptor = bf_match[0].queryIdx;
            auto keypoint = orb_keypoints[index_descriptor];
            hbst_matchable.emplace_back(
                new Matchable(keypoint, orb_descriptors.row(index_descriptor), id));
        }
    });
    hbst_matchable.shrink_to_fit();

    hbst_binary_tree_->matchAndAdd(hbst_matchable, descriptor_matches_,
                                   config_.hamming_distance_threshold,
                                   srrg_hbst::SplittingStrategy::SplitEven);
    density_maps_.emplace(id, std::move(density_map));
    std::vector<int> reference_indices;
    reference_indices.reserve(descriptor_matches_.size());
    std::for_each(descriptor_matches_.cbegin(), descriptor_matches_.cend(),
                  [&](const auto &descriptor_match) {
                      if ((id - descriptor_match.first) > 3) {
                          reference_indices.emplace_back(descriptor_match.first);
                      }
                  });
    return reference_indices;
}

ClosureCandidate MapClosures::ValidateClosure(const int reference_id, const int query_id) const {
    const Tree::MatchVector &matches = descriptor_matches_.at(reference_id);
    const size_t num_matches = matches.size();

    ClosureCandidate closure;
    if (num_matches > 2) {
        std::vector<PointPair> keypoint_pairs(num_matches);
        std::transform(matches.cbegin(), matches.cend(), keypoint_pairs.begin(),
                       [&](const Tree::Match &match) {
                           auto query_point =
                               Eigen::Vector2d(match.object_query.pt.y, match.object_query.pt.x);
                           auto ref_point = Eigen::Vector2d(match.object_references[0].pt.y,
                                                            match.object_references[0].pt.x);
                           return PointPair(ref_point, query_point);
                       });

        const auto &[pose2d, number_of_inliers] = RansacAlignment2D(keypoint_pairs);
        closure.source_id = reference_id;
        closure.target_id = query_id;
        closure.pose.block<2, 2>(0, 0) = pose2d.linear();
        closure.pose.block<2, 1>(0, 3) = pose2d.translation() * config_.density_map_resolution;
        closure.number_of_inliers = number_of_inliers;
    }
    return closure;
}
}  // namespace map_closures
