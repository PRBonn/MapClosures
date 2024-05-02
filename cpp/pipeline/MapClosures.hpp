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
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/DensityMap.hpp"
#include "srrg_hbst/types/binary_tree.hpp"

static constexpr int descriptor_size_bits = 256;
using Matchable = srrg_hbst::BinaryMatchable<cv::KeyPoint, descriptor_size_bits>;
using Node = srrg_hbst::BinaryNode<Matchable>;
using Tree = srrg_hbst::BinaryTree<Node>;

namespace map_closures {
struct Config {
    float density_map_resolution = 0.5;
    float density_threshold = 0.05;
    int hamming_distance_threshold = 50;
};

class MapClosures {
    using Matchable = srrg_hbst::BinaryMatchable<cv::KeyPoint, descriptor_size_bits>;
    using Node = srrg_hbst::BinaryNode<Matchable>;
    using Tree = srrg_hbst::BinaryTree<Node>;
    using MatchableVector = Tree::MatchableVector;

    using Points2D = std::vector<Eigen::Vector2d>;

public:
    MapClosures();
    MapClosures(const map_closures::Config &config);

    ~MapClosures() = default;

public:
    cv::Mat AddNewLocalMap(int map_idx, const std::vector<Eigen::Vector3d> &local_map);
    std::tuple<Eigen::Matrix4d, int> CheckForClosure(const int ref_idx, const int query_idx);
    std::vector<int> GetRefMapIndices(const int top_n);

private:
    void AddToTree(int map_idx, const cv::Mat &density_map_cv);
    std::tuple<bool, Points2D, Points2D, std::vector<double>> ComputeMatches(const int ref_idx,
                                                                             const int query_idx);

private:
    Config config_;
    cv::Ptr<cv::DescriptorExtractor> image_descriptor_extractor_;

    std::vector<std::unordered_map<Eigen::Vector2i, double, PixelHash>> density_maps_;
    std::vector<cv::Mat> density_maps_cv_;
    std::vector<Eigen::Vector2i> img_to_density_map_shift_;

    std::unique_ptr<Tree> hbst_tree_ = std::make_unique<Tree>();
    Tree::MatchVectorMap latest_matches_map_;
};
}  // namespace map_closures
