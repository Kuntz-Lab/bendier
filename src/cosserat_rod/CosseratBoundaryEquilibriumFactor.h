#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>

#include <optional>

class CosseratBoundaryEquilibriumFactor : public gtsam::NoiseModelFactor {
public:
    CosseratBoundaryEquilibriumFactor(
        gtsam::Key stress_key,
        std::optional<gtsam::Key> wrench_key,
        const gtsam::SharedNoiseModel& model,
        bool is_base = false);

    gtsam::Vector unwhitenedError(
        const gtsam::Values& x,
        gtsam::OptionalMatrixVecType H = nullptr) const override;

private:
    bool is_base_ = false;
};
