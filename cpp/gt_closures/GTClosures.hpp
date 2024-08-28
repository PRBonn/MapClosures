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
