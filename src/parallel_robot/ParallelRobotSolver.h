#pragma once

#include <gtsam/base/Matrix.h>

#include "parallel_robot/ParallelRobotModel.h"
#include "utils/Gaussians.h"
#include "utils/SolverBase.h"


struct ActuationForceMeas {
    gtsam::Vector6 meas;
    double sigma;
};


struct ParallelRobotSolverConfig {
    SolverBaseConfig base;
    
    int nodes_per_rod;

    gtsam::Matrix6 K_inv;

    double sigma_strain_pos;
    double sigma_strain_rot;

    double sigma_small_force;
    double sigma_small_moment;

    std::array<gtsam::Matrix4, NUM_RODS> base_end_poses;
    std::array<gtsam::Matrix4, NUM_RODS> tip_end_poses;

    double sigma_end_pose_pos;
    double sigma_end_pose_rot;
};


class ParallelRobotSolver : SolverBase {
public:
    ParallelRobotSolver(const ParallelRobotSolverConfig& config);

    Solution<ParallelRobotMarginals> solve(
        const std::array<double, NUM_RODS>& rod_lengths,
        double sigma_rod_lengths,
        const Vector6Gaussian& wrench,
        const std::optional<ActuationForceMeas>& f_meas);

private:
    void build_graph() override;

    void extract_solution() override;

    void get_initial_values() override;

    gtsam::SharedDiagonal small_wrench_noise_;
    gtsam::SharedDiagonal base_pose_noise_;

    std::unique_ptr<ParallelRobot> robot_;

    std::array<double, NUM_RODS> rod_lengths_;
    double sigma_rod_lengths_;
    Vector6Gaussian wrench_;
    std::optional<ActuationForceMeas> f_meas_;

    ParallelRobotMarginals extracted_;
};