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

using Eigen::Matrix4d;
using Eigen::Vector3d, Eigen::Vector2d;

using KeyPoints2D = std::vector<Vector2d>;

namespace {
// fixed parameters for OpenCV ORB Features
static constexpr float scale_factor = 1.00;
static constexpr int n_levels = 1;
static constexpr int first_level = 0;
static constexpr int WTA_K = 2;
static constexpr int nfeatures = 500;
static constexpr int edge_threshold = 31;
static constexpr int score_type = 1;
static constexpr int patch_size = 31;
static constexpr int fast_threshold = 35;
}  // namespace

namespace map_closures {
MapClosures::MapClosures() : config_(Config()) {
    image_descriptor_extractor_ =
        cv::ORB::create(nfeatures, scale_factor, n_levels, edge_threshold, first_level, WTA_K,
                        cv::ORB::ScoreType(score_type), patch_size, fast_threshold);
}

MapClosures::MapClosures(const Config &config) : config_(config) {
    image_descriptor_extractor_ =
        cv::ORB::create(nfeatures, scale_factor, n_levels, edge_threshold, first_level, WTA_K,
                        cv::ORB::ScoreType(score_type), patch_size, fast_threshold);
}
void MapClosures::AddToTree(int map_idx, const cv::Mat &density_map_cv) {
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> keypoints;
    image_descriptor_extractor_->detectAndCompute(density_map_cv, cv::noArray(), keypoints,
                                                  descriptors);
    auto hbst_matchable = Tree::getMatchables(descriptors, keypoints, map_idx);
    hbst_tree_->matchAndAdd(hbst_matchable, latest_matches_map_, config_.hamming_distance_threshold,
                            srrg_hbst::SplittingStrategy::SplitEven);
}

cv::Mat MapClosures::AddNewLocalMap(int map_idx, const std::vector<Vector3d> &local_map) {
    const auto density_map =
        GenerateDensityMap(local_map, config_.density_map_resolution, config_.density_threshold);
    const auto [density_map_cv, lower_bound] = DensityMapAsImage(density_map);

    density_maps_.emplace_back(density_map);
    density_maps_cv_.emplace_back(density_map_cv);
    img_to_density_map_shift_.emplace_back(lower_bound);
    this->AddToTree(map_idx, density_map_cv);
    return density_map_cv;
}

std::vector<int> MapClosures::GetRefMapIndices(const int top_n) {
    std::vector<int> ref_map_indices;
    ref_map_indices.reserve(top_n);
    if (latest_matches_map_.size() != 0) {
        std::vector<std::tuple<int, int>> num_matches_per_img;
        num_matches_per_img.reserve(latest_matches_map_.size());
        for (const auto &[ref_idx, matches] : latest_matches_map_) {
            num_matches_per_img.emplace_back(std::make_tuple(ref_idx, matches.size()));
        }
        std::sort(num_matches_per_img.begin(), num_matches_per_img.end(),
                  [](const std::tuple<int, int> &a, const std::tuple<int, int> &b) {
                      return std::get<1>(a) > std::get<1>(b);
                  });
        for (int i = 0; i < std::min(top_n, static_cast<int>(latest_matches_map_.size())); i++) {
            ref_map_indices.emplace_back(std::get<0>(num_matches_per_img[i]));
        }
    }
    return ref_map_indices;
}

std::tuple<bool, KeyPoints2D, KeyPoints2D, std::vector<double>> MapClosures::ComputeMatches(
    int ref_idx, int query_idx) {
    Tree::MatchVector matches = latest_matches_map_.at(ref_idx);
    std::sort(matches.begin(), matches.end(),
              [](const Tree::Match &a, const Tree::Match &b) { return a.distance < b.distance; });

    KeyPoints2D ref_keypoints;
    ref_keypoints.reserve(matches.size());
    KeyPoints2D query_keypoints;
    query_keypoints.reserve(matches.size());
    std::vector<double> match_weights;
    match_weights.reserve(matches.size());

    if (matches.size() >= 10) {
        auto ref_shift = img_to_density_map_shift_[ref_idx];
        auto query_shift = img_to_density_map_shift_[query_idx];

        std::for_each(matches.cbegin(), matches.cend(), [&](const Tree::Match &match) {
            if (match.object_references.size() == 1) {
                match_weights.emplace_back(match.distance);
                ref_keypoints.emplace_back(
                    Vector2d(match.object_references[0].pt.y + ref_shift.x(),
                             match.object_references[0].pt.x + ref_shift.y()));
                query_keypoints.emplace_back(Vector2d(match.object_query.pt.y + query_shift.x(),
                                                      match.object_query.pt.x + query_shift.y()));
            }
        });

        auto sum_weights = std::accumulate(match_weights.cbegin(), match_weights.cend(), 0.1);
        std::transform(match_weights.begin(), match_weights.end(), match_weights.begin(),
                       [sum_weights](auto weight) { return 1 - (weight / sum_weights); });
        return {true, ref_keypoints, query_keypoints, match_weights};
    }
    return {false, ref_keypoints, query_keypoints, match_weights};
}

std::tuple<Matrix4d, int> MapClosures::CheckForClosure(int ref_idx, int query_idx) {
    const auto [has_enough_matches, ref_kps, query_kps, match_weights] =
        this->ComputeMatches(ref_idx, query_idx);
    if (has_enough_matches) {
        const auto [R, tr, inliers_count] =
            ICPRansac2D(ref_kps, query_kps, match_weights);
        Matrix4d initial_tf;
        initial_tf.setIdentity();
        initial_tf.block(0, 0, 2, 2) = R;
        initial_tf.block(0, 3, 2, 1) = tr * config_.density_map_resolution;
        return {initial_tf, inliers_count};
    }
    return {Matrix4d{}, 0};
}
}  // namespace map_closures
