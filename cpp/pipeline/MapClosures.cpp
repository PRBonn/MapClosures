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

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <map>
#include <opencv2/core.hpp>
#include <utility>
#include <vector>

#include "core/AlignRansac2D.hpp"
#include "core/DensityMap.hpp"
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

ClosureCandidate MapClosures::MatchAndAdd(const int &id,
                                          const std::vector<Eigen::Vector3d> &local_map) {
    DensityMap density_map =
        GenerateDensityMap(local_map, config_.density_map_resolution, config_.density_threshold);
    cv::Mat orb_descriptors;
    std::vector<cv::KeyPoint> orb_keypoints;
    orb_extractor_->detectAndCompute(density_map.grid, cv::noArray(), orb_keypoints,
                                     orb_descriptors);

    auto hbst_matchable = Tree::getMatchables(orb_descriptors, orb_keypoints, id);
    hbst_binary_tree_->matchAndAdd(hbst_matchable, descriptor_matches_,
                                   config_.hamming_distance_threshold,
                                   srrg_hbst::SplittingStrategy::SplitEven);
    density_maps_.emplace(id, std::move(density_map));
    std::vector<int> indices(descriptor_matches_.size());
    std::iota(indices.begin(), indices.end(), 0);
    auto compare_closure_candidates = [](ClosureCandidate a,
                                         const ClosureCandidate &b) -> ClosureCandidate {
        return a.number_of_inliers > b.number_of_inliers ? a : b;
    };
    using iterator_type = std::vector<int>::const_iterator;
    const auto &closure = tbb::parallel_reduce(
        tbb::blocked_range<iterator_type>{indices.cbegin(), indices.cend()}, ClosureCandidate(),
        [&](const tbb::blocked_range<iterator_type> &r,
            ClosureCandidate candidate) -> ClosureCandidate {
            return std::transform_reduce(
                r.begin(), r.end(), candidate, compare_closure_candidates, [&](const auto &ref_id) {
                    const int is_far_enough = std::abs(static_cast<int>(ref_id) - id) > 3;
                    return is_far_enough ? ValidateClosure(ref_id, id) : ClosureCandidate();
                });
        },
        compare_closure_candidates);
    return closure;
}

ClosureCandidate MapClosures::ValidateClosure(const int reference_id, const int query_id) const {
    const Tree::MatchVector &matches = descriptor_matches_.at(reference_id);
    const size_t num_matches = matches.size();

    ClosureCandidate closure;
    if (num_matches > 2) {
        const auto &ref_map_lower_bound = density_maps_.at(reference_id).lower_bound;
        const auto &qry_map_lower_bound = density_maps_.at(query_id).lower_bound;
        auto to_world_point = [](const auto &p, const auto &offset) {
            return Eigen::Vector2d(p.y + offset.x(), p.x + offset.y());
        };
        std::vector<PointPair> keypoint_pairs(num_matches);
        std::transform(
            matches.cbegin(), matches.cend(), keypoint_pairs.begin(),
            [&](const Tree::Match &match) {
                auto ref_point = to_world_point(match.object_references[0].pt, ref_map_lower_bound);
                auto query_point = to_world_point(match.object_query.pt, qry_map_lower_bound);
                return PointPair(ref_point, query_point);
            });

        const auto &[T, inliers_count] = RansacAlignment2D(keypoint_pairs);
        closure.source_id = reference_id;
        closure.target_id = query_id;
        closure.T.block<2, 2>(0, 0) = T.linear();
        closure.T.block<2, 1>(0, 3) = T.translation() * config_.density_map_resolution;
        closure.number_of_inliers = inliers_count;
    }
    return closure;
}
}  // namespace map_closures
