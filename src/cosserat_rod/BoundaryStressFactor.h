#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Pose3.h>


using BoundaryStressBase = gtsam::NoiseModelFactorN<gtsam::Vector6, gtsam::Vector6, gtsam::Pose3>;

class BoundaryStressFactor: public BoundaryStressBase {
    using BoundaryStressBase::evaluateError;

public:
    BoundaryStressFactor(
        gtsam::Key stress_key,
        gtsam::Key wrench_key,
        gtsam::Key pose_key,
        const gtsam::SharedNoiseModel& model,
        bool is_base);
        
    gtsam::Vector evaluateError(
        const gtsam::Vector6& stress, 
        const gtsam::Vector6& wrench,
        const gtsam::Pose3& pose,
        gtsam::OptionalMatrixType H1, 
        gtsam::OptionalMatrixType H2,
        gtsam::OptionalMatrixType H3) const override;

private:
    bool is_base_;
};
