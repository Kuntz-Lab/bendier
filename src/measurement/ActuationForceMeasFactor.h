#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>


class ActuationForceMeasFactor: public gtsam::NoiseModelFactorN<gtsam::Vector6> {
    double f_z_meas_;

public:
    using gtsam::NoiseModelFactorN<gtsam::Vector6>::evaluateError;
  
    ActuationForceMeasFactor(
        gtsam::Key wrench_key,
        double f_z_meas,
        const gtsam::SharedNoiseModel& model);

    gtsam::Vector evaluateError(const gtsam::Vector6& wrench, gtsam::OptionalMatrixType H1) const override;
};
