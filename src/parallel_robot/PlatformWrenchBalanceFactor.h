#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

class PlatformWrenchBalanceFactor: public gtsam::NoiseModelFactor {
public:
    PlatformWrenchBalanceFactor(
        const gtsam::KeyVector& tip_wrench_keys,
        const gtsam::KeyVector& pose_keys,
        gtsam::Key platform_wrench_key,
        gtsam::Key platform_pose_key,
        const gtsam::SharedNoiseModel& model);

    gtsam::Vector unwhitenedError(
        const gtsam::Values& x,
        gtsam::OptionalMatrixVecType H = nullptr) const override;

private:
    size_t num_rods_;
};
