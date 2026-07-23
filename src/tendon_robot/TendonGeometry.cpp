#include "TendonGeometry.h"

using namespace gtsam;

Vector3 tendon_hole_diff(
    const Pose3& pose0,
    const Pose3& pose1,
    const Point3& hole0,
    const Point3& hole1,
    OptionalJacobian<3, 6> H_pose0,
    OptionalJacobian<3, 6> H_pose1)
{
    // TF hole1 (in pose1's frame) into pose0's frame for differencing.
    Matrix36 d_hole1w_d_pose1;
    Point3 hole1_world = pose1.transformFrom(hole1, d_hole1w_d_pose1);

    Matrix36 d_hole10_d_pose0;
    Matrix3 d_hole10_d_hole1w;
    Point3 hole10 = pose0.transformTo(hole1_world, d_hole10_d_pose0, d_hole10_d_hole1w);

    if (H_pose0) *H_pose0 = d_hole10_d_pose0;
    if (H_pose1) *H_pose1 = d_hole10_d_hole1w * d_hole1w_d_pose1;

    // Return difference between the holes in pose0's frame
    return hole10 - hole0;
}
