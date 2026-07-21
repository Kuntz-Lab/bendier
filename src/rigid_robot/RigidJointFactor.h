#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

enum class JointType { Revolute = 0, Prismatic = 1 };

using RigidJointBase = gtsam::NoiseModelFactorN<
    gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Vector>;

// Ties two consecutive link poses together through a joint value pulled out
// of a single shared joint-vector variable (mirrors how TendonDiscWrenchFactor
// reads one tendon's tension out of a shared `tensions` Vector), composed
// with an offset that is itself a Pose3 variable rather than a fixed
// constant. The predicted child pose is:
//
//   pose_parent * offset * Expmap(screw * q)
//
// `offset` is the joint's as-realized static transform -- a calibration
// parameter that's never truly known, so the solver puts a prior on it
// (nominal URDF origin as the mean, assembly/manufacturing tolerance as the
// covariance) rather than baking it in as a constant. `q` is this joint's
// actuated value, read out of the shared joint-vector variable, which gets
// its own separate prior. This factor's own noise model is the residual
// slack in the kinematic chain identity itself (kept tight/near-deterministic
// -- all the real uncertainty lives in the offset and joint-value priors).
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
    int joint_idx_;
    gtsam::Vector6 screw_;  // unit generalized axis: (axis,0) revolute, (0,axis) prismatic
};
