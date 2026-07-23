#pragma once

#include <gtsam/geometry/Pose3.h>

// Shared helper for computing hole difference and its Jacobians
gtsam::Vector3 tendon_hole_diff(
    const gtsam::Pose3& pose0,
    const gtsam::Pose3& pose1,
    const gtsam::Point3& hole0,
    const gtsam::Point3& hole1,
    gtsam::OptionalJacobian<3, 6> H_pose0 = {},
    gtsam::OptionalJacobian<3, 6> H_pose1 = {});
