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

// Specifies structure of a single revolute or prismatic joint.
struct RigidJointSpec {
    // Prior calibration of the parent link's frame to the child link's frame when the joint is at zero position.
    // Note that this cov also specifies the uncertainty of the joint value itself, e.g. cov(2,2) is variance of revolute angle. 
    Pose3Gaussian offset_calibration;

    // Axis of rotation or translation, expressed in the parent link's frame.
    gtsam::Vector3 axis = gtsam::Vector3::UnitZ();
    JointType type = JointType::Revolute;
};

struct RigidRobotMarginals {
    std::vector<Pose3Gaussian> links;  // World frame link poses
    std::vector<Pose3Gaussian> offsets;  // World frame calibration offsets
    VectorXGaussian joints;  // Posterior over the joint values
    Pose3Gaussian tip_pose;  // Tip is offset by a fixed calibration transform 

    // Jacobian of the tip pose wrt the joint values
    gtsam::Matrix J_tip_joints;

    // Only populated if wrench sensing is enabled.
    std::optional<Vector6Gaussian> tip_wrench;    // Posterior for the external tip wrench (world frame)
    std::optional<VectorXGaussian> joint_torques; // Posterior for the per-joint generalized force, projected from tip_wrench
};

struct RigidRobotModelConfig {
    std::vector<RigidJointSpec> joints;

    // Calibration prior on the world-frame pose of link 0
    Pose3Gaussian base_pose_calibration;

    // Calibration prior on the tip frame relative to the final link 
    Pose3Gaussian tip_offset_calibration;

    // Shared 6-dof noise on the kinematic chain identity itself
    gtsam::SharedDiagonal chain_noise;

    // Quasistatic wrench sensing (optional)
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
