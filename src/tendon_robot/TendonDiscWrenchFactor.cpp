#include "TendonDiscWrenchFactor.h"

#include <gtsam/base/Matrix.h>

#include "utils/WrenchTransforms.h"


using namespace gtsam;


TendonDiscWrenchFactor::TendonDiscWrenchFactor(
    Key pose_prev_key,
    Key pose_key,
    Key pose_next_key, // Set to dummy key if we are at the tip
    Key wrench_key,
    Key tensions_key,
    Key external_wrench_key,
    const bool is_tip,
    const std::array<Point3, NUM_TENDONS>& holes_prev,
    const std::array<Point3, NUM_TENDONS>& holes,
    const std::array<Point3, NUM_TENDONS>& holes_next, // Not used if we are at the tip
    const SharedNoiseModel& model)
: 
    TendonWrenchBase(model, pose_prev_key, pose_key, pose_next_key, wrench_key, tensions_key, external_wrench_key),
    is_tip_(is_tip),
    holes_prev_(holes_prev),
    holes_(holes),
    holes_next_(holes_next) {}


Vector TendonDiscWrenchFactor::evaluateError(
    const Pose3& pose_prev, 
    const Pose3& pose, 
    const Pose3& pose_next, 
    const Vector6& wrench, 
    const Vector4& tensions,
    const Vector6& wrench_external,
    OptionalMatrixType H1, 
    OptionalMatrixType H2, 
    OptionalMatrixType H3, 
    OptionalMatrixType H4, 
    OptionalMatrixType H5,
    OptionalMatrixType H6) const 
{
    Vector6 wrench_tendons = Vector6::Zero();
    
    Matrix64 d_wrench_d_tensions = Matrix64::Zero();
    Matrix66 d_wrench_d_pose = Matrix66::Zero();
    Matrix66 d_wrench_d_pose_prev = Matrix66::Zero();
    Matrix66 d_wrench_d_pose_next = Matrix66::Zero();

    // Sum up all tendon wrenches on this disc
    for (int tendon_idx = 0; tendon_idx < tensions.size(); ++tendon_idx) {
        // Wrench from previous disc
        Vector6 d_wrench_prev_d_tension;
        Matrix6 d_wrench_prev_d_pose, d_wrench_prev_d_pose_prev;

        // Get wrench from prev disc on current disc in spatial coords
        Vector6 wrench_prev = get_single_tendon_wrench(
            tensions[tendon_idx],
            pose,
            pose_prev,
            holes_[tendon_idx],
            holes_prev_[tendon_idx],
            d_wrench_prev_d_tension,
            d_wrench_prev_d_pose,
            d_wrench_prev_d_pose_prev);
        
        wrench_tendons += wrench_prev;
        Vector6 d_wrench_d_tension = d_wrench_prev_d_tension;
        d_wrench_d_pose += d_wrench_prev_d_pose;
        d_wrench_d_pose_prev += d_wrench_prev_d_pose_prev;
        
        // Wrench from next disc. Ignore if we are at the tip
        if (!is_tip_){
            Vector6 d_wrench_next_d_tension;
            Matrix6 d_wrench_next_d_pose, d_wrench_next_d_pose_next;

            Vector6 wrench_next = get_single_tendon_wrench(
                tensions[tendon_idx], 
                pose,
                pose_next, 
                holes_[tendon_idx],
                holes_next_[tendon_idx],
                d_wrench_next_d_tension,
                d_wrench_next_d_pose,
                d_wrench_next_d_pose_next);
            
            wrench_tendons += wrench_next;
            d_wrench_d_tension += d_wrench_next_d_tension;
            d_wrench_d_pose += d_wrench_next_d_pose;
            d_wrench_d_pose_next += d_wrench_next_d_pose_next;
        }

        d_wrench_d_tensions.col(tendon_idx) = d_wrench_d_tension;
    }

    // Error between total wrench and sum of applied, all in spatial coords
    Vector6 wrench_error = wrench - wrench_tendons - wrench_external;

    if (H1) { *H1 = -d_wrench_d_pose_prev; }

    if (H2) { *H2 = -d_wrench_d_pose; }

    if (H3) { *H3 = -d_wrench_d_pose_next; }

    if (H4) { *H4 = Matrix6::Identity(); }

    if (H5) { *H5 = -d_wrench_d_tensions; }

    if (H6) { *H6 = -Matrix6::Identity(); }

    return wrench_error;
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