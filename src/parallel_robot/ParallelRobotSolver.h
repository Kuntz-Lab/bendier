#pragma once

#include <array>
#include <optional>

#include <gtsam/base/Matrix.h>

#include "parallel_robot/ParallelRobotModel.h"
#include "utils/Gaussians.h"
#include "utils/SolverBase.h"

struct ActuationForceMeas {
    gtsam::Vector6 meas;
    double sigma;
};

struct ParallelRobotSolverConfig {
    ParallelRobotSolverConfig(
        int nodes_per_rod,
        const gtsam::Matrix6& K_inv,
        double sigma_strain_rot,
        double sigma_strain_pos,
        double sigma_small_force,
        double sigma_small_moment,
        const std::array<gtsam::Matrix4, NUM_RODS>& base_end_poses,
        const std::array<gtsam::Matrix4, NUM_RODS>& tip_end_poses,
        double sigma_end_pose_pos,
        double sigma_end_pose_rot,
        const SolverBaseConfig& base = {})
    :   base(base),
        nodes_per_rod(nodes_per_rod),
        K_inv(K_inv),
        sigma_strain_rot(sigma_strain_rot),
        sigma_strain_pos(sigma_strain_pos),
        sigma_small_force(sigma_small_force),
        sigma_small_moment(sigma_small_moment),
        base_end_poses(base_end_poses),
        tip_end_poses(tip_end_poses),
        sigma_end_pose_pos(sigma_end_pose_pos),
        sigma_end_pose_rot(sigma_end_pose_rot)
    {}

    SolverBaseConfig base;

    int nodes_per_rod;

    gtsam::Matrix6 K_inv;

    double sigma_strain_rot;
    double sigma_strain_pos;

    double sigma_small_force;
    double sigma_small_moment;

    std::array<gtsam::Matrix4, NUM_RODS> base_end_poses;
    std::array<gtsam::Matrix4, NUM_RODS> tip_end_poses;

    double sigma_end_pose_pos;
    double sigma_end_pose_rot;
};

class ParallelRobotSolver : public SolverBase<ParallelRobotModel> {
public:
    ParallelRobotSolver(const ParallelRobotSolverConfig& config);

    Solution<ParallelRobotModel::ModelMarginals> solve(
        const std::array<double, NUM_RODS>& rod_lengths,
        double                              sigma_rod_lengths,
        const Vector6Gaussian&              wrench,
        const std::optional<ActuationForceMeas>& f_meas = std::nullopt);
};
