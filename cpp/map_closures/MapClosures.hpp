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

#include "DensityMap.hpp"
#include "srrg_hbst/types/binary_tree.hpp"

static constexpr int descriptor_size_bits = 256;
using Matchable = srrg_hbst::BinaryMatchable<cv::KeyPoint, descriptor_size_bits>;
using Node = srrg_hbst::BinaryNode<Matchable>;
using Tree = srrg_hbst::BinaryTree<Node>;

namespace map_closures {
struct Config {
    float density_map_resolution = 0.5f;
    float density_threshold = 0.05f;
    int hamming_distance_threshold = 50;
};

struct ClosureCandidate {
    int source_id = -1;
    int target_id = -1;
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    std::size_t number_of_inliers = 0;
};

class MapClosures {
public:
    explicit MapClosures();
    explicit MapClosures(const Config &config);
    ~MapClosures() = default;

    std::vector<int> MatchAndAdd(const int id, const std::vector<Eigen::Vector3d> &local_map);
    ClosureCandidate ValidateClosure(const int reference_id, const int query_id) const;

    const DensityMap &getDensityMapFromId(const int &map_id) const {
        return density_maps_.at(map_id);
    }

private:
    Config config_;
    Tree::MatchVectorMap descriptor_matches_;
    std::unordered_map<int, DensityMap> density_maps_;
    std::unordered_map<int, Eigen::Matrix4d> ground_alignments_;
    std::unique_ptr<Tree> hbst_binary_tree_ = std::make_unique<Tree>();
    cv::Ptr<cv::DescriptorExtractor> orb_extractor_;
};
}  // namespace map_closures
