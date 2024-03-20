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

std::pair<std::vector<int>, cv::Mat> MapClosures::MatchAndAddLocalMap(
    const int map_idx,
    const std::vector<Eigen::Vector3d> &local_map,
    const long unsigned int top_k) {
    density_maps_.emplace(map_idx, GenerateDensityMap(local_map, config_.density_map_resolution,
                                                      config_.density_threshold));

    cv::Mat orb_descriptors;
    std::vector<cv::KeyPoint> orb_keypoints;
    const auto &last_density_grid = density_maps_.at(map_idx).grid;
    orb_extractor_->detectAndCompute(last_density_grid, cv::noArray(), orb_keypoints,
                                     orb_descriptors);

    auto hbst_matchable = Tree::getMatchables(orb_descriptors, orb_keypoints, map_idx);
    hbst_binary_tree_->matchAndAdd(hbst_matchable, descriptor_matches_,
                                   config_.hamming_distance_threshold,
                                   srrg_hbst::SplittingStrategy::SplitEven);

    const size_t clipped_top_k = std::min(top_k, descriptor_matches_.size());
    std::vector<int> ref_mapclosure_indices(clipped_top_k);
    if (clipped_top_k) {
        using NumMatches = size_t;
        using MapIdx = int;
        std::multimap<NumMatches, MapIdx> num_matches_per_ref_map;
        std::for_each(descriptor_matches_.cbegin(), descriptor_matches_.cend(),
                      [&](const auto &matches) {
                          num_matches_per_ref_map.emplace(matches.second.size(), matches.first);
                      });

        std::transform(num_matches_per_ref_map.crbegin(),
                       std::next(num_matches_per_ref_map.crbegin(), clipped_top_k),
                       ref_mapclosure_indices.begin(),
                       [&](const auto &num_matches_kv) { return num_matches_kv.second; });
    }
    return {ref_mapclosure_indices, last_density_grid};
}

std::pair<Eigen::Matrix4d, int> MapClosures::CheckForClosure(int ref_idx, int query_idx) const {
    const Tree::MatchVector &matches = descriptor_matches_.at(ref_idx);

    const size_t num_matches = matches.size();
    std::vector<PointPair> keypoint_pairs;
    keypoint_pairs.reserve(num_matches);

    const auto &ref_map_lower_bound = density_maps_.at(ref_idx).lower_bound;
    const auto &qry_map_lower_bound = density_maps_.at(query_idx).lower_bound;
    std::for_each(matches.cbegin(), matches.cend(), [&](const Tree::Match &match) {
        if (match.object_references.size() == 1) {
            auto ref_match = match.object_references[0].pt;
            auto qry_match = match.object_query.pt;
            keypoint_pairs.emplace_back(Eigen::Vector2d(ref_match.y + ref_map_lower_bound.x(),
                                                        ref_match.x + ref_map_lower_bound.y()),
                                        Eigen::Vector2d(qry_match.y + qry_map_lower_bound.x(),
                                                        qry_match.x + qry_map_lower_bound.y()));
        }
    });

    int num_inliers = 0;
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    if (num_matches > 2) {
        const auto &[T, inliers_count] = RansacAlignment2D(keypoint_pairs);
        pose.block<2, 2>(0, 0) = T.linear();
        pose.block<2, 1>(0, 3) = T.translation() * config_.density_map_resolution;
        num_inliers = inliers_count;
    }
    return {pose, num_inliers};
}
}  // namespace map_closures
