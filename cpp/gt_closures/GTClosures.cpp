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

#include "GTClosures.hpp"

#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_reduce.h>

#include <Eigen/Core>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include "VoxelHashSet.hpp"

namespace gt_closures {

GTClosures::GTClosures(const int dataset_size,
                       const double sampling_distance,
                       const double overlap_threshold,
                       double voxel_size,
                       const double max_range) {
    sampling_distance_ = sampling_distance;
    overlap_threshold_ = overlap_threshold;
    voxel_size_ = voxel_size;
    max_range_ = max_range;
    n_skip_segments_ = static_cast<int>(2 * max_range_ / sampling_distance_);

    poses_.reserve(dataset_size);
    scan_occupancies_.reserve(dataset_size);
    dataset_indices_.resize(dataset_size);
    std::iota(dataset_indices_.begin(), dataset_indices_.end(), 0);
}

void GTClosures::AddPointCloud(const int scan_idx,
                               const std::vector<Eigen::Vector3d> &pointcloud,
                               const Eigen::Matrix4d &pose) {
    poses_.insert({scan_idx, pose});
    auto scan_occupancy = VoxelHashSet(voxel_size_);
    scan_occupancy.AddPoints(pointcloud, pose.block<3, 3>(0, 0), pose.block<3, 1>(0, 3));
    scan_occupancies_.insert({scan_idx, scan_occupancy});
}

int GTClosures::GetSegments() {
    double traveled_distance = 0.0;
    Eigen::Matrix4d last_pose = poses_.at(0);

    int segment_idx = 0;
    std::vector<int> segment_indices;
    VoxelHashSet segment_occupancy(voxel_size_);

    std::for_each(dataset_indices_.begin(), dataset_indices_.end(), [&](const int idx) {
        traveled_distance +=
            (last_pose.block<3, 1>(0, 3) - poses_.at(idx).block<3, 1>(0, 3)).norm();
        if (traveled_distance > sampling_distance_) {
            segment_indices.shrink_to_fit();
            segments_.insert({segment_idx, {segment_indices, segment_occupancy}});
            segments_indices_.emplace_back(segment_idx);
            traveled_distance = 0.0;
            segment_indices.clear();
            segment_occupancy.clear();
            segment_idx++;
        }
        segment_indices.emplace_back(idx);
        segment_occupancy.AddVoxels(scan_occupancies_.at(idx));
        last_pose = poses_.at(idx);
        scan_occupancies_.erase(idx);
    });
    return segments_.size() - n_skip_segments_;
}

std::vector<Eigen::Vector2i> GTClosures::ComputeClosuresForQuerySegment(
    const int query_segment_idx) {
    auto &[query_segment, query_segment_occupancy] = segments_.at(query_segment_idx);
    auto query_seqment_pose = poses_.at(query_segment[0]);

    Closures closures;
    closures.reserve(query_segment.size() * poses_.size());
    closures = tbb::parallel_reduce(
        tbb::blocked_range<std::vector<int>::const_iterator>{
            segments_indices_.cbegin() + query_segment_idx + n_skip_segments_,
            segments_indices_.cend()},
        closures,
        // Transform
        [&](const tbb::blocked_range<std::vector<int>::const_iterator> &r,
            Closures closure) -> Closures {
            std::for_each(r.begin(), r.end(), [&](const int ref_segment_idx) {
                auto &[ref_segment, ref_segment_occupancy] = segments_.at(ref_segment_idx);
                auto ref_segment_pose = poses_.at(ref_segment[0]);
                auto distance =
                    (query_seqment_pose.block<3, 1>(0, 3) - ref_segment_pose.block<3, 1>(0, 3))
                        .norm();
                if (distance < max_range_) {
                    auto overlap_3d = query_segment_occupancy.ComputeOverlap(ref_segment_occupancy);
                    if (overlap_3d > overlap_threshold_) {
                        std::for_each(
                            ref_segment.cbegin(), ref_segment.cend(), [&](const int ref_id) {
                                std::for_each(
                                    query_segment.cbegin(), query_segment.cend(),
                                    [&](const int query_id) {
                                        closure.emplace_back(Eigen::Vector2i(query_id, ref_id));
                                    });
                            });
                    }
                }
            });
            return closure;
        },
        // Reduce
        [](Closures lhs, const Closures &rhs) -> Closures {
            if (!rhs.empty()) {
                lhs.insert(lhs.end(), std::make_move_iterator(rhs.cbegin()),
                           std::make_move_iterator(rhs.cend()));
            }
            return lhs;
        });
    closures.shrink_to_fit();
    segments_.erase(query_segment_idx);
    return closures;
}
}  // namespace gt_closures
