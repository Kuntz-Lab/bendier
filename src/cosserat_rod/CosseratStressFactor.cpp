#include "CosseratStressFactor.h"

#include "utils/WrenchTransforms.h"

using namespace gtsam;


CosseratStressFactor::CosseratStressFactor(
    Key pose_0_key,
    Key pose_1_key,
    Key stress_0_key,
    Key stress_1_key,
    Key wrench_key,
    const SharedNoiseModel& model)
:
    CosseratStressBase(model, pose_0_key, pose_1_key, stress_0_key, stress_1_key, wrench_key) {}


Vector CosseratStressFactor::evaluateError(
    const Pose3& p0, 
    const Pose3& p1, 
    const Vector6& s0, 
    const Vector6& s1, 
    const Vector6& w1,
    OptionalMatrixType H1, 
    OptionalMatrixType H2, 
    OptionalMatrixType H3, 
    OptionalMatrixType H4,
    OptionalMatrixType H5) const 
{
    // This factor assumes wrench is in spatial frame, must convert coordinates to body (pose_1) frame
    Matrix6 d_body_d_p1, d_body_d_w1;
    Vector6 body = spatial_to_body_wrench(w1, p1, d_body_d_w1, d_body_d_p1);

    // We transform stress_0 to pose_1 frame for summation with wrench_body
    Matrix6 d_s1_pred_d_p0, d_s1_pred_d_p1, d_s1_pred_d_s0;
    Vector6 s1_pred = transform_wrench_adjoint(
        s0, 
        p0, 
        p1, 
        d_s1_pred_d_s0,
        d_s1_pred_d_p0,
        d_s1_pred_d_p1) - body;
    
    Vector6 stress_error = s1_pred - s1;

    if (H1) { *H1 = d_s1_pred_d_p0; }

    if (H2) { *H2 = d_s1_pred_d_p1 - d_body_d_p1; }

    if (H3) { *H3 = d_s1_pred_d_s0; }
    
    if (H4) { *H4 = -Matrix6::Identity(); }

    if (H5) { *H5 = -d_body_d_w1; }

    return stress_error;
}
