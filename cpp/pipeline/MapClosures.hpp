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

#include "core/DensityMap.hpp"
#include "srrg_hbst/types/binary_tree.hpp"

namespace {
static constexpr int descriptor_size_bits = 256;

using Matchable = srrg_hbst::BinaryMatchable<cv::KeyPoint, descriptor_size_bits>;
using Node = srrg_hbst::BinaryNode<Matchable>;
using Tree = srrg_hbst::BinaryTree<Node>;
}  // namespace

namespace map_closures {
struct Config {
    float density_map_resolution = 0.5;
    float density_threshold = 0.05;
    int hamming_distance_threshold = 50;
};

class MapClosures {
public:
    explicit MapClosures();
    explicit MapClosures(const Config &config);
    ~MapClosures() = default;

public:
    std::pair<std::vector<int>, cv::Mat> MatchAndAddLocalMap(
        const int map_idx,
        const std::vector<Eigen::Vector3d> &local_map,
        const long unsigned int top_k);
    std::pair<Eigen::Matrix4d, int> CheckForClosure(const int ref_idx, const int query_idx) const;

private:
    Config config_;
    Tree::MatchVectorMap descriptor_matches_;
    std::unordered_map<int, DensityMap> density_maps_;
    std::unique_ptr<Tree> hbst_binary_tree_ = std::make_unique<Tree>();
    cv::Ptr<cv::DescriptorExtractor> orb_extractor_;
};
}  // namespace map_closures
