#include "TendonDiscWrenchFactor.h"

#include <gtsam/base/Matrix.h>

#include "utils/WrenchTransforms.h"

using namespace gtsam;

namespace {

KeyVector make_keys(
    Key pose_prev_key, Key pose_key, std::optional<Key> pose_next_key,
    Key wrench_key, Key tensions_key,
    std::optional<Key> external_wrench_key)
{
    KeyVector keys{pose_prev_key, pose_key};
    if (pose_next_key) keys.push_back(*pose_next_key);
    keys.push_back(wrench_key);
    keys.push_back(tensions_key);
    if (external_wrench_key) keys.push_back(*external_wrench_key);
    return keys;
}

}  // namespace

TendonDiscWrenchFactor::TendonDiscWrenchFactor(
    Key pose_prev_key, Key pose_key, std::optional<Key> pose_next_key,
    Key wrench_key, Key tensions_key,
    std::optional<Key> external_wrench_key,
    const std::vector<Point3>& holes_prev,
    const std::vector<Point3>& holes,
    const std::vector<Point3>& holes_next,
    const SharedNoiseModel& model)
:
    NoiseModelFactor(model, make_keys(
        pose_prev_key, pose_key, pose_next_key, wrench_key, tensions_key, external_wrench_key)),
    is_tip_(!pose_next_key.has_value()),
    has_external_wrench_(external_wrench_key.has_value()),
    holes_prev_(holes_prev),
    holes_(holes),
    holes_next_(is_tip_ ? std::vector<Point3>{} : holes_next)
{}

Vector TendonDiscWrenchFactor::unwhitenedError(
    const Values& x, OptionalMatrixVecType H) const
{
    // Key layout:
    //   Non-tip, with ext: [0]=pose_prev, [1]=pose, [2]=pose_next, [3]=wrench, [4]=tensions, [5]=ext_wrench
    //   Non-tip, no ext:   [0]=pose_prev, [1]=pose, [2]=pose_next, [3]=wrench, [4]=tensions
    //   Tip, with ext:     [0]=pose_prev, [1]=pose,                [2]=wrench, [3]=tensions, [4]=ext_wrench
    //   Tip, no ext:       [0]=pose_prev, [1]=pose,                [2]=wrench, [3]=tensions
    const int wrench_idx     = is_tip_ ? 2 : 3;
    const int tensions_idx   = wrench_idx + 1;
    const int ext_wrench_idx = tensions_idx + 1;  // only valid when has_external_wrench_

    const Pose3   pose_prev = x.at<Pose3>(keys()[0]);
    const Pose3   pose      = x.at<Pose3>(keys()[1]);
    const Pose3   pose_next = is_tip_ ? Pose3{} : x.at<Pose3>(keys()[2]);
    const Vector6 wrench    = x.at<Vector6>(keys()[wrench_idx]);
    const Vector  tensions  = x.at<Vector>(keys()[tensions_idx]);

    Vector6 wrench_tendons        = Vector6::Zero();
    Matrix  d_wrench_d_tensions   = Matrix::Zero(6, tensions.size());
    Matrix66 d_wrench_d_pose      = Matrix66::Zero();
    Matrix66 d_wrench_d_pose_prev = Matrix66::Zero();
    Matrix66 d_wrench_d_pose_next = Matrix66::Zero();

    for (int i = 0; i < tensions.size(); ++i) {
        Vector6 dT_prev;
        Matrix6 dP_prev, dPprev;

        Vector6 w_prev = get_single_tendon_wrench(
            tensions[i], pose, pose_prev,
            holes_[i], holes_prev_[i],
            dT_prev, dP_prev, dPprev);

        wrench_tendons += w_prev;
        Vector6 dT = dT_prev;
        d_wrench_d_pose      += dP_prev;
        d_wrench_d_pose_prev += dPprev;

        if (!is_tip_) {
            Vector6 dT_next;
            Matrix6 dP_next, dPnext;

            Vector6 w_next = get_single_tendon_wrench(
                tensions[i], pose, pose_next,
                holes_[i], holes_next_[i],
                dT_next, dP_next, dPnext);

            wrench_tendons += w_next;
            dT                   += dT_next;
            d_wrench_d_pose      += dP_next;
            d_wrench_d_pose_next += dPnext;
        }

        d_wrench_d_tensions.col(i) = dT;
    }

    Vector6 wrench_external = Vector6::Zero();
    if (has_external_wrench_)
        wrench_external = x.at<Vector6>(keys()[ext_wrench_idx]);

    if (H) {
        (*H)[0] = -d_wrench_d_pose_prev;
        (*H)[1] = -d_wrench_d_pose;
        if (!is_tip_) (*H)[2] = -d_wrench_d_pose_next;
        (*H)[wrench_idx]   =  Matrix6::Identity();
        (*H)[tensions_idx] = -d_wrench_d_tensions;
        if (has_external_wrench_) (*H)[ext_wrench_idx] = -Matrix6::Identity();
    }

    return wrench - wrench_tendons - wrench_external;
}

Vector6 TendonDiscWrenchFactor::get_single_tendon_wrench(
    const double tension,
    const Pose3& p0,
    const Pose3& p1,
    const Point3& h0,
    const Point3& h1,
    OptionalJacobian<6, 1> H_tension,
    OptionalJacobian<6, 6> H_p0,
    OptionalJacobian<6, 6> H_p1) const
{
    // TF body hole 1 location to frame 0 for differencing
    Matrix36 d_h1w_d_p1;
    Point3 h1w = p1.transformFrom(h1, d_h1w_d_p1);

    Matrix36 d_h10_d_p0;
    Matrix3 d_h10_d_h1w;
    Point3 h10 = p0.transformTo(h1w, d_h10_d_p0, d_h10_d_h1w); // Hole 1 in frame 0

    // Difference between two holes is direction of force
    Vector3 diff = h10 - h0; // Both holes in frame 0
    Matrix3 d_diff_d_h10 = Matrix3::Identity();

    Matrix3 d_dir_d_diff;
    Vector3 dir = normalize(diff, &d_dir_d_diff); // Frame 0

    // Force is tension in that direction
    Vector3 force = tension * dir;  // Frame 0
    Matrix31 d_force_d_tension = dir;
    Matrix33 d_force_d_dir = tension * Matrix3::Identity();

    // Compute moment about frame 0 origin and combine to wrench
    Matrix3 d_moment_d_force;
    Vector3 moment = cross(h0, force, std::nullopt, d_moment_d_force); // Frame 0

    Vector6 body;
    body << moment, force;
    Matrix63 d_body_d_moment = Matrix63::Zero();
    d_body_d_moment.topRows(3) = Matrix3::Identity();
    Matrix63 d_body_d_force = Matrix63::Zero();
    d_body_d_force.bottomRows(3) = Matrix3::Identity();

    // Our wrenches are all defined in spatial coordinates, so rotate
    Matrix6 d_spatial_d_body, d_spatial_d_p0;
    Vector6 spatial = body_to_spatial_wrench(body, p0, d_spatial_d_body, d_spatial_d_p0);

    if (H_tension) {
        *H_tension = d_spatial_d_body * d_body_d_force * d_force_d_tension +
            d_spatial_d_body * d_body_d_moment * d_moment_d_force * d_force_d_tension;
    }

    if (H_p0) {
        Matrix36 d_force_d_p0 = d_force_d_dir * d_dir_d_diff * d_diff_d_h10 * d_h10_d_p0;
        *H_p0 = d_spatial_d_p0 +
            d_spatial_d_body * d_body_d_force * d_force_d_p0 +
            d_spatial_d_body * d_body_d_moment * d_moment_d_force * d_force_d_p0;
    }

    if (H_p1) {
        Matrix36 d_force_d_p1 = d_force_d_dir * d_dir_d_diff * d_diff_d_h10 * d_h10_d_h1w * d_h1w_d_p1;
        *H_p1 = d_spatial_d_body * d_body_d_force * d_force_d_p1 +
            d_spatial_d_body * d_body_d_moment * d_moment_d_force * d_force_d_p1;
    }

    return spatial;
}
