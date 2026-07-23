#pragma once

#include <gtsam/base/Vector.h>
#include <optional>

#include "rigid_robot/RigidRobotModel.h"
#include "utils/Gaussians.h"
#include "utils/SolverBase.h"

struct RigidRobotSolverConfig {
    RigidRobotSolverConfig(
        std::vector<RigidJointSpec> joints,
        Pose3Gaussian base_pose_calibration,
        Pose3Gaussian tip_offset_calibration,
        double sigma_chain_rot = 1e-6,
        double sigma_chain_pos = 1e-6,
        bool enable_wrench_sensing = false,
        const SolverBaseConfig& base = {})
    :   base(base),
        joints(std::move(joints)),
        base_pose_calibration(std::move(base_pose_calibration)),
        tip_offset_calibration(std::move(tip_offset_calibration)),
        sigma_chain_rot(sigma_chain_rot),
        sigma_chain_pos(sigma_chain_pos),
        enable_wrench_sensing(enable_wrench_sensing)
    {}

    SolverBaseConfig base;

    std::vector<RigidJointSpec> joints;
    Pose3Gaussian base_pose_calibration;
    Pose3Gaussian tip_offset_calibration;

    double sigma_chain_rot;
    double sigma_chain_pos;

    bool enable_wrench_sensing;
};

class RigidRobotSolver : public SolverBase<RigidRobotModel> {
public:
    explicit RigidRobotSolver(const RigidRobotSolverConfig& config);

    // Solve given priors on any of these quantities, often just with single joint vector prior
    // TODO: why is there an "enable_wrench_sensing" option in the solver config if the wrench prior is optional here? Shouldn't it be required if enabled?
    Solution<RigidRobotModel::ModelMarginals> solve(
        const VectorXGaussian& joint_prior,
        const std::optional<Vector6Gaussian>& tip_wrench_prior = std::nullopt,
        const std::optional<VectorXGaussian>& joint_torque_meas = std::nullopt,
        const std::optional<Pose3Gaussian>& tip_pose_prior = std::nullopt);
};
