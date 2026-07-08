#pragma once

#include "cosserat_rod/CosseratRodModel.h"
#include "utils/Gaussians.h"
#include <gtsam/linear/NoiseModel.h>

#include <vector>

struct ParallelRobotMarginals {
    std::vector<CosseratRodMarginals> rods;

    Pose3Gaussian platform_pose;
    Vector6Gaussian platform_wrench;

    gtsam::Matrix rod_lengths_jacobian;  // 6 x num_rods
    gtsam::Matrix6 tip_wrench_jacobian;
};

class ParallelRobotModel {
public:
    using ModelMarginals = ParallelRobotMarginals;

    // Rod count is base_end_poses.size() -- must match tip_end_poses.size().
    ParallelRobotModel(
        int nodes_per_rod,
        gtsam::Matrix6 K_inv,
        gtsam::SharedDiagonal strain_noise,
        gtsam::SharedDiagonal stress_noise,
        std::vector<gtsam::Matrix4> base_end_poses,
        std::vector<gtsam::Matrix4> tip_end_poses,
        double sigma_end_pose_pos,
        double sigma_end_pose_rot,
        double sigma_rod_lengths);

    void set_rod_lengths(const gtsam::Vector& rod_lengths);

    void set_sigma_rod_lengths(double sigma_rod_lengths);

    gtsam::NonlinearFactorGraph build_graph() const;

    gtsam::Values get_initial_values() const;

    ParallelRobotMarginals get_marginals(
        const gtsam::Values& values,
        const gtsam::Marginals& marginals) const;

    gtsam::Matrix get_rod_lengths_jacobian(const gtsam::Marginals& marginals) const;  // 6 x num_rods

    gtsam::Matrix6 get_tip_wrench_jacobian(const gtsam::Marginals& marginals) const;

    gtsam::Key get_rod_wrench_key(int rod_idx, int node_idx) const;

    gtsam::Key platform_pose_key() const;

    gtsam::Key platform_wrench_key() const;

    inline int get_num_rods() const { return static_cast<int>(rods_.size()); }

private:
    std::vector<std::unique_ptr<CosseratRodModel>> rods_;
    const std::vector<gtsam::Matrix4> base_end_poses_;
    const std::vector<gtsam::Matrix4> tip_end_poses_;
    const gtsam::SharedDiagonal small_wrench_noise_;

    const int id_;
    inline static int next_id_ = 0;

    const double sigma_end_pose_pos_;
    const double sigma_end_pose_rot_;
    double sigma_rod_lengths_;
    gtsam::Vector rod_lengths_;
};
