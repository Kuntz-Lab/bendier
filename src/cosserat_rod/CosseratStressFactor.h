#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>


using CosseratStressBase = gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3, gtsam::Vector6, gtsam::Vector6, gtsam::Vector6>;

class CosseratStressFactor: public CosseratStressBase {
    using CosseratStressBase::evaluateError;

public:
    CosseratStressFactor(
        gtsam::Key pose_0_key,
        gtsam::Key pose_1_key,
        gtsam::Key stress_0_key,
        gtsam::Key stress_1_key,
        gtsam::Key wrench_key,
        const gtsam::SharedNoiseModel& model);

    gtsam::Vector evaluateError(
        const gtsam::Pose3& pose_0, 
        const gtsam::Pose3& pose_1, 
        const gtsam::Vector6& stress_0, 
        const gtsam::Vector6& stress_1, 
        const gtsam::Vector6& wrench_1,
        gtsam::OptionalMatrixType H1, 
        gtsam::OptionalMatrixType H2, 
        gtsam::OptionalMatrixType H3, 
        gtsam::OptionalMatrixType H4,
        gtsam::OptionalMatrixType H5) const override;
};
