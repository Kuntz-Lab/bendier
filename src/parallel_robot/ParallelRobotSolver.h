#pragma once

#include <optional>
#include <vector>

#include <gtsam/base/Matrix.h>

#include "parallel_robot/ParallelRobotModel.h"
#include "utils/Gaussians.h"
#include "utils/SolverBase.h"

struct ActuationForceMeas {
    gtsam::Vector meas;  // one scalar per rod
    double sigma;
};

struct ParallelRobotSolverConfig {
    ParallelRobotSolverConfig(
        int nodes_per_rod,
        const gtsam::Matrix6& K_inv,
        double sigma_constitutive_rot,
        double sigma_constitutive_pos,
        double sigma_equilibrium_force,
        double sigma_equilibrium_moment,
        std::vector<gtsam::Matrix4> base_end_poses,
        std::vector<gtsam::Matrix4> tip_end_poses,
        double sigma_end_pose_pos,
        double sigma_end_pose_rot,
        const SolverBaseConfig& base = {})
    :   base(base),
        nodes_per_rod(nodes_per_rod),
        K_inv(K_inv),
        sigma_constitutive_rot(sigma_constitutive_rot),
        sigma_constitutive_pos(sigma_constitutive_pos),
        sigma_equilibrium_force(sigma_equilibrium_force),
        sigma_equilibrium_moment(sigma_equilibrium_moment),
        base_end_poses(std::move(base_end_poses)),
        tip_end_poses(std::move(tip_end_poses)),
        sigma_end_pose_pos(sigma_end_pose_pos),
        sigma_end_pose_rot(sigma_end_pose_rot)
    {}

    SolverBaseConfig base;

    int nodes_per_rod;

    gtsam::Matrix6 K_inv;

    double sigma_constitutive_rot;
    double sigma_constitutive_pos;

    double sigma_equilibrium_force;
    double sigma_equilibrium_moment;

    std::vector<gtsam::Matrix4> base_end_poses;
    std::vector<gtsam::Matrix4> tip_end_poses;

    double sigma_end_pose_pos;
    double sigma_end_pose_rot;
};

class ParallelRobotSolver : public SolverBase<ParallelRobotModel> {
public:
    ParallelRobotSolver(const ParallelRobotSolverConfig& config);

    Solution<ParallelRobotModel::ModelMarginals> solve(
        const gtsam::Vector&                 rod_lengths,
        double                              sigma_rod_lengths,
        const Vector6Gaussian&              wrench,
        const std::optional<ActuationForceMeas>& f_meas = std::nullopt);
};
