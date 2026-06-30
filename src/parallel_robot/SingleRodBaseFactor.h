#pragma once

#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

class SingleRodBaseFactor: public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Vector6> {
    using gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Vector6>::evaluateError;

public:
    SingleRodBaseFactor(
        gtsam::Key pose_key, 
        gtsam::Key stress_key,
        const gtsam::Pose3& pose,
        const gtsam::SharedNoiseModel& model);
        
    gtsam::Vector evaluateError(
        const gtsam::Pose3& pose,
        const gtsam::Vector6& stress,
        gtsam::OptionalMatrixType H1,
        gtsam::OptionalMatrixType H2) const override;

private:
    const gtsam::Pose3 pose_;
};
