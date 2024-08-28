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
#include <unordered_map>
#include <vector>

#include "VoxelHashMap.hpp"

using Closures = std::vector<Eigen::Vector2i>;

namespace gt_closures {

struct Segment {
    std::vector<int> indices;
    VoxelHashMap map;
};

class GTClosures {
public:
    GTClosures(const int dataset_size,
               const double sampling_distance,
               const double overlap_threshold,
               double overlap_voxel_size,
               const double max_range);

    void AddPointCloud(const int idx,
                       const std::vector<Eigen::Vector3d> &pointcloud,
                       const Eigen::Matrix4d &pose);
    int GetSegments();
    std::vector<Eigen::Vector2i> ComputeClosuresForQuerySegment(const int query_segment_idx);

private:
    std::vector<int> dataset_indices_;
    std::unordered_map<int, Eigen::Matrix4d> poses_;
    std::unordered_map<int, std::vector<Eigen::Vector3d>> pointclouds_;

    std::vector<int> segments_indices_;
    std::unordered_map<int, Segment> segments_;

    double sampling_distance_ = 2.0;
    double overlap_threshold_ = 0.5;
    double overlap_voxel_size_ = 0.5;
    double max_range_ = 100.0;
    int n_skip_segments_ = 0;
};
}  // namespace gt_closures
