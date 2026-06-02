#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

class PositionPriorFactor: public gtsam::NoiseModelFactorN<gtsam::Pose3> {
    gtsam::Vector3 position_meas_;

public:
    using gtsam::NoiseModelFactorN<gtsam::Pose3>::evaluateError;
  
    PositionPriorFactor(
        gtsam::Key pose_key,
        gtsam::Vector3 position_meas,
        const gtsam::SharedNoiseModel& model);

    gtsam::Vector evaluateError(const gtsam::Pose3& pose, gtsam::OptionalMatrixType H1) const override;
};
