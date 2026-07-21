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
        double sigma_chain_rot = 1e-6,
        double sigma_chain_pos = 1e-6,
        bool enable_wrench_sensing = false,
        const SolverBaseConfig& base = {})
    :   base(base),
        joints(std::move(joints)),
        base_pose_calibration(std::move(base_pose_calibration)),
        sigma_chain_rot(sigma_chain_rot),
        sigma_chain_pos(sigma_chain_pos),
        enable_wrench_sensing(enable_wrench_sensing)
    {}

    SolverBaseConfig base;

    std::vector<RigidJointSpec> joints;
    Pose3Gaussian base_pose_calibration;

    double sigma_chain_rot;
    double sigma_chain_pos;

    bool enable_wrench_sensing;
};

class RigidRobotSolver : public SolverBase<RigidRobotModel> {
public:
    explicit RigidRobotSolver(const RigidRobotSolverConfig& config);

    // tip_wrench_prior: a direct prior/measurement on the external wrench at
    // the tip (e.g. a wrist force/torque sensor).
    //
    // joint_torque_meas: a prior/measurement on the full joint-torque
    // vector (e.g. from motor current/effort sensors), mirroring how
    // joint_prior works for joint values -- one RigidJointTorqueFactor is
    // added per joint, using sqrt(joint_torque_meas.cov(i,i)) as that
    // joint's own sensor noise (off-diagonal/cross-joint sensor correlation
    // isn't modeled).
    //
    // Either or both can be given; each contributes independent information
    // about the same single tip-wrench variable (see RigidRobotModel) --
    // this is what lets joint-torque measurements alone recover a full 6-dof
    // tip wrench estimate, the same way a 7-DOF arm's redundancy shows up
    // elsewhere. Passing neither (with wrench sensing enabled) leaves the
    // tip wrench fully unconstrained.
    Solution<RigidRobotModel::ModelMarginals> solve(
        const VectorXGaussian& joint_prior,
        const std::optional<Vector6Gaussian>& tip_wrench_prior = std::nullopt,
        const std::optional<VectorXGaussian>& joint_torque_meas = std::nullopt);
};
