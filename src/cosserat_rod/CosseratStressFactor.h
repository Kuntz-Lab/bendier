#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

class CosseratStressFactor : public gtsam::NoiseModelFactor {
public:
    // Interior element: 5 keys
    CosseratStressFactor(
        gtsam::Key pose_0_key,
        gtsam::Key pose_1_key,
        gtsam::Key stress_0_key,
        gtsam::Key stress_1_key,
        gtsam::Key wrench_key,
        const gtsam::SharedNoiseModel& model);

    // Tip element: 4 keys
    CosseratStressFactor(
        gtsam::Key pose_0_key,
        gtsam::Key pose_1_key,
        gtsam::Key stress_0_key,
        gtsam::Key stress_1_key,
        const gtsam::SharedNoiseModel& model);

    gtsam::Vector unwhitenedError(
        const gtsam::Values& x,
        gtsam::OptionalMatrixVecType H = nullptr) const override;
};
