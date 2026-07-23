#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

enum class JointType { Revolute = 0, Prismatic = 1 };

using RigidJointBase = gtsam::NoiseModelFactorN<
    gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Vector>;

class RigidJointFactor : public RigidJointBase {
    using RigidJointBase::evaluateError;

public:
    RigidJointFactor(
        gtsam::Key pose_parent_key,
        gtsam::Key pose_child_key,
        gtsam::Key offset_key,
        gtsam::Key joint_vector_key,
        int joint_idx,
        const gtsam::Vector3& axis,
        JointType type,
        const gtsam::SharedNoiseModel& model);

    gtsam::Vector evaluateError(
        const gtsam::Pose3& pose_parent,
        const gtsam::Pose3& pose_child,
        const gtsam::Pose3& offset,
        const gtsam::Vector& joint_vec,
        gtsam::OptionalMatrixType H1,
        gtsam::OptionalMatrixType H2,
        gtsam::OptionalMatrixType H3,
        gtsam::OptionalMatrixType H4) const override;

private:
    // What robot joint/link index is this?
    int joint_idx_;
    
    // Screw axis of the joint, expressed in the parent link's frame
    gtsam::Vector6 screw_;
};
