#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>

#include <optional>
#include <vector>

#include "RigidJointFactor.h"
#include "utils/Gaussians.h"

// One joint of a serial kinematic chain. `offset_calibration` is the prior
// on the fixed transform from the parent link's frame to this joint's
// unactuated child frame (e.g. a URDF joint's <origin>) -- its mean is that
// nominal transform, and its covariance is the "uncertainty in each
// component" of the joint's realized offset (assembly tolerance, backlash,
// structural compliance -- a calibration parameter that's never truly
// known). `axis` and `type` say how the joint value moves the child frame
// away from that offset. See RigidJointFactor.
struct RigidJointSpec {
    Pose3Gaussian offset_calibration;
    gtsam::Vector3 axis = gtsam::Vector3::UnitZ();
    JointType type = JointType::Revolute;
};

struct RigidLinkState {
    Pose3Gaussian pose;
};

struct RigidRobotMarginals {
    std::vector<RigidLinkState> links;    // size num_links, world-frame link poses (actuated chain only)
    std::vector<Pose3Gaussian> offsets;   // size num_joints, per-joint calibration offset posterior
    VectorXGaussian joints;               // posterior over the full joint-value vector

    // The end-effector/tool-tip pose -- links.back() plus the fixed
    // tip_offset_calibration below. Always populated; this is what a
    // tip-pose prior/gizmo should target, and what wrench sensing treats
    // as the tip.
    Pose3Gaussian tip_pose;

    // Jacobian of the tip pose (tangent space at tip_pose.mean) with
    // respect to the joint-value vector, 6 x num_joints. For this 7-DOF
    // (or any redundant) arm, its null space is the "self-motion" direction
    // that moves the joints without moving the tip -- same idea as
    // TendonRobotMarginals.J_pose_tensions, computed the same way (from the
    // tip-pose/joint-vector joint marginal), just with joints playing the
    // role tensions did there.
    gtsam::Matrix J_tip_joints;

    // Only populated if wrench sensing is enabled.
    std::optional<Vector6Gaussian> tip_wrench;    // posterior over the external tip wrench (world frame)
    std::optional<VectorXGaussian> joint_torques; // per-joint generalized force, projected from tip_wrench
};

struct RigidRobotModelConfig {
    std::vector<RigidJointSpec> joints;

    // Calibration prior on the world-frame pose of link 0 -- standardized
    // the same way as each joint's offset_calibration above (mean + full
    // 6-dof covariance), rather than being a separate special case.
    Pose3Gaussian base_pose_calibration;

    // Calibration on the fixed transform from the last actuated link's
    // frame to the actual end-effector/tool-tip frame (e.g. a URDF's
    // trailing fixed joints out to the flange, which this model otherwise
    // has no reason to know about since it only tracks actuated joints).
    // No joint motion is involved, so this is just a BetweenFactor between
    // the last link's pose and a new tip_pose variable -- no separate
    // offset variable needed (there's nothing else to tie it to).
    Pose3Gaussian tip_offset_calibration;

    // Shared 6-dof noise on the kinematic chain identity itself
    // (pose_child == pose_parent * offset * Expmap(screw*q)). Kept tight --
    // the real uncertainty lives in each offset_calibration and the
    // joint-value prior.
    gtsam::SharedDiagonal chain_noise;

    // Quasistatic wrench sensing (optional): when true, adds a single
    // Vector6 world-frame tip-wrench variable. There's no per-link wrench
    // state to add beyond that -- with no distributed load along a rigid
    // link (no gravity, no mid-link forces), the wrench transmitted across
    // every joint is just the tip wrench transported to that joint's
    // location (see RigidJointTorqueFactor), so a single variable is all the
    // physics needs, the same way the joint-value vector is a single
    // variable rather than one per link. Off by default so the
    // pose-estimation-only use case pays nothing for it -- and because with
    // it on, the solver must supply enough of a tip-wrench prior and/or
    // joint-torque measurements to pin down the otherwise fully free
    // tip wrench (see RigidRobotSolver::solve).
    bool enable_wrench_sensing = false;
};

class RigidRobotModel {
public:
    using ModelMarginals = RigidRobotMarginals;
    explicit RigidRobotModel(const RigidRobotModelConfig& config);

    gtsam::NonlinearFactorGraph build_graph() const;

    gtsam::Values get_initial_values(
        const std::optional<gtsam::Pose3>& base_pose_init = std::nullopt,
        const gtsam::Vector& nominal_joint_values = gtsam::Vector()) const;

    RigidRobotMarginals get_marginals(
        const gtsam::Values& values,
        const gtsam::Marginals& marginals) const;

    inline int get_num_links() const { return num_links_; }
    inline int get_num_joints() const { return num_joints_; }
    inline bool has_wrench_sensing() const { return enable_wrench_sensing_; }

    gtsam::Key get_pose_key(int link_idx) const;
    gtsam::Key get_offset_key(int joint_idx) const;
    gtsam::Key get_joint_vector_key() const;
    gtsam::Key get_tip_wrench_key() const;
    gtsam::Key get_tip_pose_key() const;

    const gtsam::Vector3& get_joint_axis(int joint_idx) const;
    JointType get_joint_type(int joint_idx) const;

    const std::vector<gtsam::Key>& get_pose_keys() const;

private:
    int clamp_link_idx(int link_idx) const;
    int clamp_joint_idx(int joint_idx) const;

    // Unique robot ID for unique GTSAM Symbol keys
    const int id_;
    inline static int next_id_ = 0;

    const int num_joints_;
    const int num_links_;

    std::vector<RigidJointSpec> joint_specs_;
    Pose3Gaussian base_pose_calibration_;
    Pose3Gaussian tip_offset_calibration_;
    gtsam::SharedDiagonal chain_noise_;

    const bool enable_wrench_sensing_;

    std::vector<gtsam::Key> pose_keys_;
    std::vector<gtsam::Key> offset_keys_;
};
