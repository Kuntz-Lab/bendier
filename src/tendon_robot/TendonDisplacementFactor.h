#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <vector>

class TendonDisplacementFactor : public gtsam::NoiseModelFactor {
public:
    TendonDisplacementFactor(
        std::vector<gtsam::Key> disc_pose_keys,
        gtsam::Key tensions_key,
        gtsam::Key displacements_key,
        std::vector<std::vector<gtsam::Point3>> hole_locations,  // [disc][tendon]
        std::vector<double> reference_lengths,                    // [tendon]
        double tendon_stiffness,
        const gtsam::SharedNoiseModel& model);

    gtsam::Vector unwhitenedError(
        const gtsam::Values& x,
        gtsam::OptionalMatrixVecType H = nullptr) const override;

private:
    int num_discs_;
    std::vector<std::vector<gtsam::Point3>> hole_locations_;
    std::vector<double> reference_lengths_;
    double tendon_stiffness_;
};
