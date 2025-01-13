#include <open3d/Open3D.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "map_closures/GroundAlign.hpp"
#include "map_closures/MapClosures.hpp"

namespace fs = std::filesystem;

void TransformPoints(const Eigen::Matrix4d &T, std::vector<Eigen::Vector3d> &pointcloud) {
    auto R = T.block<3, 3>(0, 0);
    auto t = T.block<3, 1>(0, 3);
    std::transform(pointcloud.cbegin(), pointcloud.cend(), pointcloud.begin(),
                   [&](const auto &point) { return R * point + t; });
}

int main(int argc, char *argv[]) {
    const fs::path datadir{argv[1]};
    std::vector<fs::path> local_map_files;
    for (const auto &dir_entry : fs::directory_iterator(datadir)) {
        local_map_files.emplace_back(dir_entry.path());
    }
    std::sort(local_map_files.begin(), local_map_files.end(), [](fs::path a, fs::path b) {
        return std::stoi(a.stem().string()) < std::stoi(b.stem().string());
    });

    auto pipeline = map_closures::MapClosures(map_closures::Config());

    int map_id = 0;
    std::for_each(local_map_files.cbegin(), local_map_files.cend(), [&](const auto &filename) {
        open3d::geometry::PointCloud pointcloud;
        open3d::io::ReadPointCloudFromPLY(filename, pointcloud, open3d::io::ReadPointCloudOption());

        auto points = pointcloud.points_;
        auto T_ground = map_closures::AlignToLocalGround(points, 5.0);
        TransformPoints(T_ground, points);
        auto closure = pipeline.MatchAndAdd(map_id, points);
        map_id++;
    });
    return 0;
}
