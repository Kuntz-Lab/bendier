#include "RigidJointTorqueFactor.h"
#include "utils/WrenchTransforms.h"

using namespace gtsam;

RigidJointTorqueFactor::RigidJointTorqueFactor(
    Key pose_tip_key,
    Key pose_child_key,
    Key tip_wrench_key,
    const Vector3& axis,
    JointType type,
    double torque_meas,
    const SharedNoiseModel& model)
:
    RigidJointTorqueBase(model, pose_tip_key, pose_child_key, tip_wrench_key),
    axis_(axis),
    type_(type),
    torque_meas_(torque_meas)
{}

Vector RigidJointTorqueFactor::evaluateError(
    const Pose3& pose_tip,
    const Pose3& pose_child,
    const Vector6& tip_wrench,
    OptionalMatrixType H1,
    OptionalMatrixType H2,
    OptionalMatrixType H3) const
{
    // No distributed load along a rigid link -- the wrench at every joint is
    // just the tip wrench, transported to that joint's own location.
    Matrix6 H_wrench, H_pose_tip, H_pose_child;
    Vector6 transported = transform_wrench_translation(
        tip_wrench, pose_tip, pose_child, H_wrench, H_pose_tip, H_pose_child);

    Matrix36 d_rot_d_pose_child;
    Rot3 rot = pose_child.rotation(d_rot_d_pose_child);

    Matrix3 d_axis_world_d_rot;
    Vector3 axis_world = rot.rotate(axis_, d_axis_world_d_rot);

    bool revolute = (type_ == JointType::Revolute);
    Vector3 component = revolute ? Vector3(transported.head<3>()) : Vector3(transported.tail<3>());

    Vector1 error;
    error(0) = axis_world.dot(component) - torque_meas_;

    // d(tau)/d(transported): axis_world dotted into whichever half
    // (moment/force) `component` was selected from.
    Matrix16 d_tau_d_transported = Matrix16::Zero();
    if (revolute) d_tau_d_transported.block<1,3>(0,0) = axis_world.transpose();
    else          d_tau_d_transported.block<1,3>(0,3) = axis_world.transpose();

    if (H1) *H1 = d_tau_d_transported * H_pose_tip;

    if (H2) {
        // pose_child affects tau two ways: through the transport target
        // point, and through the axis direction itself.
        Matrix16 d_tau_d_pose_child_via_axis = component.transpose() * d_axis_world_d_rot * d_rot_d_pose_child;
        *H2 = d_tau_d_transported * H_pose_child + d_tau_d_pose_child_via_axis;
    }

    if (H3) *H3 = d_tau_d_transported * H_wrench;

    return error;
}
