#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

using PlatformWrenchBase = gtsam::NoiseModelFactorN<
    gtsam::Vector6, gtsam::Pose3,
    gtsam::Vector6, gtsam::Pose3,
    gtsam::Vector6, gtsam::Pose3,
    gtsam::Vector6, gtsam::Pose3,
    gtsam::Vector6, gtsam::Pose3,
    gtsam::Vector6, gtsam::Pose3,
    gtsam::Vector6, gtsam::Pose3>;

class PlatformWrenchBalanceFactor: public PlatformWrenchBase {
    using PlatformWrenchBase::evaluateError;

public:
    PlatformWrenchBalanceFactor(
        gtsam::Key stress_key_0, gtsam::Key pose_key_0,
        gtsam::Key stress_key_1, gtsam::Key pose_key_1,
        gtsam::Key stress_key_2, gtsam::Key pose_key_2,
        gtsam::Key stress_key_3, gtsam::Key pose_key_3,
        gtsam::Key stress_key_4, gtsam::Key pose_key_4,
        gtsam::Key stress_key_5, gtsam::Key pose_key_5,
        gtsam::Key platform_wrench_key, gtsam::Key platform_pose_key,
        const gtsam::SharedNoiseModel& model);
        
    gtsam::Vector evaluateError(
        const gtsam::Vector6& stress_0, const gtsam::Pose3& pose_0,
        const gtsam::Vector6& stress_1, const gtsam::Pose3& pose_1,
        const gtsam::Vector6& stress_2, const gtsam::Pose3& pose_2,
        const gtsam::Vector6& stress_3, const gtsam::Pose3& pose_3,
        const gtsam::Vector6& stress_4, const gtsam::Pose3& pose_4,
        const gtsam::Vector6& stress_5, const gtsam::Pose3& pose_5,
        const gtsam::Vector6& platform_wrench, const gtsam::Pose3& platform_pose,
        gtsam::OptionalMatrixType H1, gtsam::OptionalMatrixType H2,
        gtsam::OptionalMatrixType H3, gtsam::OptionalMatrixType H4, 
        gtsam::OptionalMatrixType H5, gtsam::OptionalMatrixType H6,
        gtsam::OptionalMatrixType H7, gtsam::OptionalMatrixType H8,
        gtsam::OptionalMatrixType H9, gtsam::OptionalMatrixType H10, 
        gtsam::OptionalMatrixType H11, gtsam::OptionalMatrixType H12,
        gtsam::OptionalMatrixType H13, gtsam::OptionalMatrixType H14) const override;
};
