#include "BoundaryStressFactor.h"

#include "utils/WrenchTransforms.h"

using namespace gtsam;

BoundaryStressFactor::BoundaryStressFactor(
    Key stress_key,
    Key wrench_key,
    Key pose_key,
    const SharedNoiseModel& model,
    bool is_base)
:
    is_base_(is_base),
    BoundaryStressBase(model, stress_key, wrench_key, pose_key) 
{}

Vector BoundaryStressFactor::evaluateError(
    const Vector6& stress, 
    const Vector6& wrench,
    const Pose3& pose,
    OptionalMatrixType H1, 
    OptionalMatrixType H2,
    OptionalMatrixType H3) const 
{
    // This factor assumes wrench is in spatial frame, must convert coordinates to body (pose_0) frame
    Matrix6 d_wrench_body_d_pose, d_wrench_body_d_wrench;
    Vector6 wrench_body = spatial_to_body_wrench(wrench, pose, d_wrench_body_d_wrench, d_wrench_body_d_pose);

    // At the base, the stress is negative wrench, since it flows out of the rod
    double sign = is_base_ ? 1.0 : -1.0;

    Vector6 stress_error = stress + sign * wrench_body;

    if (H1) { *H1 = Matrix6::Identity(); }

    if (H2) { *H2 = sign * d_wrench_body_d_wrench; }

    if (H3) { *H3 = sign * d_wrench_body_d_pose; }

    return stress_error;
}
