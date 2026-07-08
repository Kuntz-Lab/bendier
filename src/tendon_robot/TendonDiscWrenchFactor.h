#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <optional>
#include <vector>

#include "tendon_robot/TendonRobotModel.h"

class TendonDiscWrenchFactor : public gtsam::NoiseModelFactor {
public:
    TendonDiscWrenchFactor(
        gtsam::Key pose_prev_key,
        gtsam::Key pose_key,
        std::optional<gtsam::Key> pose_next_key,
        gtsam::Key wrench_key,
        gtsam::Key tensions_key,
        std::optional<gtsam::Key> external_wrench_key,
        const std::vector<gtsam::Point3>& holes_prev,
        const std::vector<gtsam::Point3>& holes,
        const std::vector<gtsam::Point3>& holes_next,
        const gtsam::SharedNoiseModel& model);

    gtsam::Vector unwhitenedError(
        const gtsam::Values& x,
        gtsam::OptionalMatrixVecType H = nullptr) const override;

private:
    gtsam::Vector6 get_single_tendon_wrench(
        const double tension,
        const gtsam::Pose3& pose,
        const gtsam::Pose3& pose_other,
        const gtsam::Point3& hole,
        const gtsam::Point3& hole_other,
        gtsam::OptionalJacobian<6, 1> H_tension = {},
        gtsam::OptionalJacobian<6, 6> H_pose = {},
        gtsam::OptionalJacobian<6, 6> H_pose_other = {}) const;

    bool is_tip_;
    bool has_external_wrench_;
    std::vector<gtsam::Point3> holes_prev_;
    std::vector<gtsam::Point3> holes_;
    std::vector<gtsam::Point3> holes_next_;  // unused when is_tip_
};
