#include "TendonDisplacementFactor.h"

#include <gtsam/base/Matrix.h>

#include "TendonGeometry.h"

using namespace gtsam;

namespace {

KeyVector make_keys(
    const std::vector<Key>& disc_pose_keys, Key tensions_key, Key displacements_key)
{
    KeyVector keys(disc_pose_keys.begin(), disc_pose_keys.end());
    keys.push_back(tensions_key);
    keys.push_back(displacements_key);
    return keys;
}

}  // namespace

TendonDisplacementFactor::TendonDisplacementFactor(
    std::vector<Key> disc_pose_keys,
    Key tensions_key,
    Key displacements_key,
    std::vector<std::vector<Point3>> hole_locations,
    std::vector<double> reference_lengths,
    std::vector<double> axial_stiffness,
    const SharedNoiseModel& model)
:
    NoiseModelFactor(model, make_keys(disc_pose_keys, tensions_key, displacements_key)),
    num_discs_(static_cast<int>(disc_pose_keys.size())),
    hole_locations_(std::move(hole_locations)),
    reference_lengths_(std::move(reference_lengths)),
    axial_stiffness_(std::move(axial_stiffness))
{}

Vector TendonDisplacementFactor::unwhitenedError(
    const Values& x, OptionalMatrixVecType H) const
{
    const int num_tendons = static_cast<int>(reference_lengths_.size());

    std::vector<Pose3> poses(num_discs_);
    for (int k = 0; k < num_discs_; ++k) poses[k] = x.at<Pose3>(keys()[k]);

    const int tensions_idx     = num_discs_;
    const int displacements_idx = num_discs_ + 1;

    const Vector tensions     = x.at<Vector>(keys()[tensions_idx]);
    const Vector displacements = x.at<Vector>(keys()[displacements_idx]);

    // Each element of error is the predicted tendon displacement minus the real displacement
    Vector error(num_tendons);

    std::vector<Matrix> d_error_d_pose(num_discs_, Matrix::Zero(num_tendons, 6));
    Matrix d_error_d_tensions = Matrix::Zero(num_tendons, num_tendons);

    // Loop over tendons to fill error/Jacobian rows
    for (int i = 0; i < num_tendons; ++i) {

        // Sum of geometric lengths
        double l_geom = 0.0;

        // Loop over discs to sum all tendon segment lengths
        for (int k = 0; k + 1 < num_discs_; ++k) {
            // Difference between two holes 
            Matrix36 d_diff_d_pose_k, d_diff_d_pose_k1;
            Vector3 diff = tendon_hole_diff(
                poses[k], poses[k + 1],
                hole_locations_[k][i], hole_locations_[k + 1][i],
                d_diff_d_pose_k, d_diff_d_pose_k1);

            double seg_len = diff.norm();  // TODO gtsam should have a built in jacobian for this 
            l_geom += seg_len;

            Vector3 dir = diff / seg_len;
            d_error_d_pose[k].row(i)     += dir.transpose() * d_diff_d_pose_k;
            d_error_d_pose[k + 1].row(i) += dir.transpose() * d_diff_d_pose_k1;
        }

        // Elastic stretch uses the reference length rather than the current geometric length
        // This should be a negligible second order effect
        double stretch_coeff = reference_lengths_[i] / axial_stiffness_[i];

        // Predicted displacement based on elasticity theory and current robot shape
        error(i) = displacements(i) - reference_lengths_[i] + l_geom - tensions(i) * stretch_coeff;

        d_error_d_tensions(i, i) = -stretch_coeff;
    }

    if (H) {
        for (int k = 0; k < num_discs_; ++k) (*H)[k] = d_error_d_pose[k];
        (*H)[tensions_idx]      = d_error_d_tensions;
        (*H)[displacements_idx] = Matrix::Identity(num_tendons, num_tendons);
    }

    return error;
}
