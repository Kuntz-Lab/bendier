#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include "RigidJointFactor.h"

using RigidJointTorqueBase = gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3, gtsam::Vector6>;

class RigidJointTorqueFactor : public RigidJointTorqueBase {
public:
    using RigidJointTorqueBase::evaluateError;

    RigidJointTorqueFactor(
        gtsam::Key pose_tip_key,
        gtsam::Key pose_child_key,
        gtsam::Key tip_wrench_key,
        const gtsam::Vector3& axis,
        JointType type,
        double torque_meas,
        const gtsam::SharedNoiseModel& model);

    gtsam::Vector evaluateError(
        const gtsam::Pose3& pose_tip,
        const gtsam::Pose3& pose_child,
        const gtsam::Vector6& tip_wrench,
        gtsam::OptionalMatrixType H1,
        gtsam::OptionalMatrixType H2,
        gtsam::OptionalMatrixType H3) const override;

private:
    gtsam::Vector3 axis_;
    JointType type_;
    double torque_meas_;
};
