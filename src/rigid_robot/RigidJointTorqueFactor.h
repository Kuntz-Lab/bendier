#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include "RigidJointFactor.h"  // for JointType

using RigidJointTorqueBase = gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3, gtsam::Vector6>;

// Measurement factor for a joint torque/effort sensor, under a quasistatic
// assumption: with no distributed load along any rigid link (no gravity, no
// mid-link forces), the wrench transmitted across *every* joint is simply
// the external tip wrench, transported to that joint's own location --
// there's no need for a separate per-link wrench variable/chain (a rigid
// link, unlike an elastic rod, carries no independent internal state).
//
// The generalized force driving a joint is the projection of that
// transported wrench onto the joint's own motion axis: for a revolute joint
// that's the moment component along the (world-frame) axis (tau = axis .
// moment, the standard "moment of a wrench about an axis" -- equivalent to
// axis . (r x F) for a pure force F applied at lever arm r from the axis);
// for a prismatic joint it's the force component along the axis instead.
//
// The joint's world-frame axis is read from pose_child's rotation. This is
// valid for both joint types: rotating a frame about its own axis (revolute)
// or translating along it (prismatic) never changes that axis's own
// direction, so pose_child's orientation always agrees with the joint's own
// (unactuated) frame on the axis direction -- no need to carry the offset
// variable into this factor at all.
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
