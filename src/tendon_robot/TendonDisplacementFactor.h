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
        std::vector<double> axial_stiffness,                      // [tendon] // TODO axis stiffness doesn't need to be a vector for now
        const gtsam::SharedNoiseModel& model);

    gtsam::Vector unwhitenedError(
        const gtsam::Values& x,
        gtsam::OptionalMatrixVecType H = nullptr) const override;

private:
    int num_discs_;
    std::vector<std::vector<gtsam::Point3>> hole_locations_;
    std::vector<double> reference_lengths_;
    std::vector<double> axial_stiffness_;  // TODO just call this "tendon_stiffness" or "k_tendon" or something
};
