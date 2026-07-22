#include "RigidJointFactor.h"

#include <gtsam/base/Matrix.h>

using namespace gtsam;

RigidJointFactor::RigidJointFactor(
    Key pose_parent_key,
    Key pose_child_key,
    Key offset_key,
    Key joint_vector_key,
    int joint_idx,
    const Vector3& axis,
    JointType type,
    const SharedNoiseModel& model)
:
    RigidJointBase(model, pose_parent_key, pose_child_key, offset_key, joint_vector_key),
    joint_idx_(joint_idx)
{
    screw_.setZero();
    if (type == JointType::Revolute) screw_.head<3>() = axis;
    else                             screw_.tail<3>() = axis;
}

Vector RigidJointFactor::evaluateError(
    const Pose3& pose_parent,
    const Pose3& pose_child,
    const Pose3& offset,
    const Vector& joint_vec,
    OptionalMatrixType H1,
    OptionalMatrixType H2,
    OptionalMatrixType H3,
    OptionalMatrixType H4) const
{
    double q = joint_vec(joint_idx_);
    Vector6 xi = screw_ * q;

    Matrix6 H_offset_self, H_offset_xi;
    Pose3 T_offset = offset.retract(xi, H_offset_self, H_offset_xi);

    Matrix6 H_pred_parent, H_pred_offset;
    Pose3 T_pred = pose_parent.compose(T_offset, H_pred_parent, H_pred_offset);

    Matrix6 H_error_pred, H_error_child;
    Vector6 error = T_pred.localCoordinates(pose_child, H_error_pred, H_error_child);

    if (H1) *H1 = H_error_pred * H_pred_parent;
    if (H2) *H2 = H_error_child;
    if (H3) *H3 = H_error_pred * H_pred_offset * H_offset_self;

    if (H4) {
        Vector6 d_error_d_q = H_error_pred * H_pred_offset * H_offset_xi * screw_;

        Matrix d_error_d_joint = Matrix::Zero(6, joint_vec.size());
        d_error_d_joint.col(joint_idx_) = d_error_d_q;
        *H4 = d_error_d_joint;
    }

    return error;
}
