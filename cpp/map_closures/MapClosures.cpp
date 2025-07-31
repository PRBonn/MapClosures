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
#include <numeric>
#include <opencv2/core.hpp>
#include <utility>
#include <vector>

#include "AlignRansac2D.hpp"
#include "DensityMap.hpp"
#include "GroundAlign.hpp"
#include "srrg_hbst/types/binary_tree.hpp"

namespace {
static constexpr int min_no_of_matches = 2;
static constexpr int no_of_local_maps_to_skip = 3;
static constexpr double ground_alignment_resolution = 5.0;
static constexpr double self_similarity_threshold = 35.0;

// fixed parameters for OpenCV ORB Features
static constexpr float scale_factor = 1.00f;
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

void MapClosures::MatchAndAddToDatabase(const int id,
                                        const std::vector<Eigen::Vector3d> &local_map) {
    const Eigen::Matrix4d &T_ground = AlignToLocalGround(local_map, ground_alignment_resolution);
    DensityMap density_map = GenerateDensityMap(local_map, T_ground, config_.density_map_resolution,
                                                config_.density_threshold);
    cv::Mat orb_descriptors;
    std::vector<cv::KeyPoint> orb_keypoints;
    orb_keypoints.reserve(nfeatures);
    orb_extractor_->detectAndCompute(density_map.grid, cv::noArray(), orb_keypoints,
                                     orb_descriptors);
    orb_keypoints.shrink_to_fit();

    const auto self_matcher = cv::BFMatcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> self_matches;
    self_matches.reserve(orb_keypoints.size());
    self_matcher.knnMatch(orb_descriptors, orb_descriptors, self_matches, 2);

    density_maps_.emplace(id, std::move(density_map));
    ground_alignments_.emplace(id, T_ground);

    std::vector<Matchable *> hbst_matchable;
    hbst_matchable.reserve(orb_descriptors.rows);
    std::for_each(self_matches.cbegin(), self_matches.cend(), [&](const auto &self_match) {
        if (self_match[1].distance > self_similarity_threshold) {
            const auto index_descriptor = self_match[0].queryIdx;
            cv::KeyPoint keypoint = orb_keypoints[index_descriptor];
            keypoint.pt.x = keypoint.pt.x + static_cast<float>(density_map.lower_bound.y());
            keypoint.pt.y = keypoint.pt.y + static_cast<float>(density_map.lower_bound.x());
            hbst_matchable.emplace_back(
                new Matchable(keypoint, orb_descriptors.row(index_descriptor), id));
        }
    });
    hbst_matchable.shrink_to_fit();

    hbst_binary_tree_->matchAndAdd(hbst_matchable, descriptor_matches_,
                                   config_.hamming_distance_threshold,
                                   srrg_hbst::SplittingStrategy::SplitEven);
}

ClosureCandidate MapClosures::ValidateClosure(const int reference_id, const int query_id) const {
    const Tree::MatchVector &matches = descriptor_matches_.at(reference_id);
    const size_t num_matches = matches.size();

    ClosureCandidate closure;
    if (num_matches > min_no_of_matches) {
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
        closure.pose = ground_alignments_.at(query_id).inverse() * closure.pose *
                       ground_alignments_.at(reference_id);
        closure.number_of_inliers = number_of_inliers;
    }
    return closure;
}

ClosureCandidate MapClosures::GetBestClosure(const int query_id,
                                             const std::vector<Eigen::Vector3d> &local_map) {
    MatchAndAddToDatabase(query_id, local_map);
    auto compare_closure_candidates = [](ClosureCandidate a,
                                         const ClosureCandidate &b) -> ClosureCandidate {
        return a.number_of_inliers > b.number_of_inliers ? a : b;
    };
    auto is_far_enough = [](const int ref_id, const int query_id) {
        return std::abs(query_id - ref_id) > no_of_local_maps_to_skip;
    };
    const auto &closure = std::transform_reduce(
        descriptor_matches_.cbegin(), descriptor_matches_.cend(), ClosureCandidate(),
        compare_closure_candidates, [&](const auto &descriptor_match) {
            const auto ref_id = static_cast<int>(descriptor_match.first);
            return is_far_enough(ref_id, query_id) ? ValidateClosure(ref_id, query_id)
                                                   : ClosureCandidate();
        });
    return closure;
}

std::vector<ClosureCandidate> MapClosures::GetTopKClosures(
    const int query_id, const std::vector<Eigen::Vector3d> &local_map, const int k) {
    MatchAndAddToDatabase(query_id, local_map);
    auto compare_closure_candidates = [](const ClosureCandidate &a, const ClosureCandidate &b) {
        return a.number_of_inliers >= b.number_of_inliers;
    };
    auto is_far_enough = [](const int ref_id, const int query_id) {
        return std::abs(query_id - ref_id) > no_of_local_maps_to_skip;
    };

    std::vector<ClosureCandidate> closures;
    closures.reserve(query_id);
    std::for_each(descriptor_matches_.cbegin(), descriptor_matches_.cend(),
                  [&](const auto &descriptor_match) {
                      const auto ref_id = static_cast<int>(descriptor_match.first);
                      if (is_far_enough(ref_id, query_id)) {
                          const ClosureCandidate &closure = ValidateClosure(ref_id, query_id);
                          if (closure.number_of_inliers > min_no_of_matches) {
                              closures.emplace_back(closure);
                          }
                      }
                  });
    closures.shrink_to_fit();

    if (k != -1) {
        std::sort(closures.begin(), closures.end(), compare_closure_candidates);
        closures.resize(std::min(k, static_cast<int>(closures.size())));
    }
    return closures;
}
}  // namespace map_closures
