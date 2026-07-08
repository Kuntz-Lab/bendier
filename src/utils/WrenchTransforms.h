#pragma once

#include <gtsam/geometry/Pose3.h>

gtsam::Vector6 transform_wrench_translation(
        const gtsam::Vector6& wrench_0,
        const gtsam::Pose3& pose_0,
        const gtsam::Pose3& pose,
        gtsam::OptionalJacobian<6, 6> H_wrench_0 = {},
        gtsam::OptionalJacobian<6, 6> H_pose_0 = {},
        gtsam::OptionalJacobian<6, 6> H_pose = {});

gtsam::Vector6 spatial_to_body_wrench(
        const gtsam::Vector6& wrench_spatial, 
        const gtsam::Pose3& pose, 
        gtsam::OptionalJacobian<6, 6> H_wrench = {},
        gtsam::OptionalJacobian<6, 6> H_pose = {});

gtsam::Vector6 body_to_spatial_wrench(
        const gtsam::Vector6& wrench_body, 
        const gtsam::Pose3& pose, 
        gtsam::OptionalJacobian<6, 6> H_wrench = {},
        gtsam::OptionalJacobian<6, 6> H_pose = {});
