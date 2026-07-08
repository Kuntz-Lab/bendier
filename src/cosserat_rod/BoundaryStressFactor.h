#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>

using BoundaryStressBase = gtsam::NoiseModelFactorN<gtsam::Vector6, gtsam::Vector6>;

class BoundaryStressFactor: public BoundaryStressBase {
    using BoundaryStressBase::evaluateError;

public:
    BoundaryStressFactor(
        gtsam::Key stress_key,
        gtsam::Key wrench_key,
        const gtsam::SharedNoiseModel& model,
        bool is_base);

    gtsam::Vector evaluateError(
        const gtsam::Vector6& stress,
        const gtsam::Vector6& wrench,
        gtsam::OptionalMatrixType H1,
        gtsam::OptionalMatrixType H2) const override;

private:
    bool is_base_;
};
