#include "CosseratStrainFactor.h"
#include "utils/WrenchTransforms.h"
#include <gtsam/base/Matrix.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

using namespace gtsam;

CosseratStrainFactor::CosseratStrainFactor(
    Key pose_0_key,
    Key pose_1_key,
    Key stress_0_key,
    Key stress_1_key,
    double ds,
    const Vector6& nominal_strain,
    const Matrix6& K_inv,
    const SharedNoiseModel& model,
    int num_magnus_terms)
:
    CosseratStrainBase(model, pose_0_key, pose_1_key, stress_0_key, stress_1_key),
    ds_(ds),
    nominal_strain_(nominal_strain),
    num_magnus_terms_(num_magnus_terms),
    K_inv_(K_inv) 
{}

static Vector6 get_strain_magnus(
    const Vector6& w0,
    const Vector6& w1,
    const double ds,
    const int num_terms,
    OptionalJacobian<6,6> H0,
    OptionalJacobian<6,6> H1)
{
    if (num_terms < 1 || num_terms > 4)
        throw std::invalid_argument("num_terms must be 1..4");

    Vector6 dw = w1 - w0;
    Matrix6 d_dw_d_w0 = -Matrix6::Identity();
    Matrix6 d_dw_d_w1 = Matrix6::Identity();

    // Arrays for terms and their derivatives
    std::array<Vector6, 4> terms;
    std::array<Matrix6, 4> d_terms_d_w0;
    std::array<Matrix6, 4> d_terms_d_w1;

    // Term 0 (Euler)
    terms[0] = w0;
    d_terms_d_w0[0] = Matrix6::Identity();
    d_terms_d_w1[0] = Matrix6::Zero();

    // Term 1 (Midpoint)
    terms[1] = 0.5 * dw;
    d_terms_d_w0[1] = 0.5 * d_dw_d_w0;
    d_terms_d_w1[1] = 0.5 * d_dw_d_w1;

    // Term 2
    double c = -ds / 12.0;
    Matrix6 d_ad1_d_dw, d_ad1_d_w0;
    Vector6 ad1 = Pose3::adjoint(dw, w0, d_ad1_d_dw, d_ad1_d_w0);

    Matrix6 d_ad1_d_w0_total = d_ad1_d_w0 + d_ad1_d_dw * d_dw_d_w0;
    Matrix6 d_ad1_d_w1_total = d_ad1_d_dw * d_dw_d_w1;

    terms[2] = c * ad1;
    d_terms_d_w0[2] = c * d_ad1_d_w0_total;
    d_terms_d_w1[2] = c * d_ad1_d_w1_total;

    // Term 3
    double d = ds * ds / 240.0;
    Matrix6 d_ad2_d_dw, d_ad2_d_ad1;
    Vector6 ad2 = Pose3::adjoint(dw, ad1, d_ad2_d_dw, d_ad2_d_ad1);

    Matrix6 d_ad2_d_w0_total = d_ad2_d_dw * d_dw_d_w0 + d_ad2_d_ad1 * d_ad1_d_w0_total;
    Matrix6 d_ad2_d_w1_total = d_ad2_d_dw * d_dw_d_w1 + d_ad2_d_ad1 * d_ad1_d_w1_total;

    terms[3] = d * ad2;
    d_terms_d_w0[3] = d * d_ad2_d_w0_total;
    d_terms_d_w1[3] = d * d_ad2_d_w1_total;

    // Accumulate only up to requested num_terms
    Vector6 strain = Vector6::Zero();
    Matrix6 J0 = Matrix6::Zero();
    Matrix6 J1 = Matrix6::Zero();
    for (int i = 0; i < num_terms; ++i) {
        strain += terms[i];
        J0 += d_terms_d_w0[i];
        J1 += d_terms_d_w1[i];
    }

    if (H0) *H0 = J0;
    if (H1) *H1 = J1;

    return strain;
}

Vector CosseratStrainFactor::evaluateError(
    const Pose3& p0, 
    const Pose3& p1, 
    const Vector6& s0, 
    const Vector6& s1, 
    OptionalMatrixType H1,
    OptionalMatrixType H2, 
    OptionalMatrixType H3, 
    OptionalMatrixType H4) const 
{
    // Twist of p1 relative to p0, inverse of pose1 = pose0 * exp(ds * strain)
    Matrix6 d_twist_d_p0, d_twist_d_p1;
    Vector6 twist = p0.localCoordinates(p1, d_twist_d_p0, d_twist_d_p1);

    // Stresses at endpoints are world-frame, so rotate into each node's own body frame first.
    Matrix6 d_s0body_d_s0, d_s0body_d_p0;
    Vector6 s0_body = spatial_to_body_wrench(s0, p0, d_s0body_d_s0, d_s0body_d_p0);

    Matrix6 d_s1body_d_s1, d_s1body_d_p1;
    Vector6 s1_body = spatial_to_body_wrench(s1, p1, d_s1body_d_s1, d_s1body_d_p1);

    // Convert body stresses to strains using constitutive law (linear elasticity)
    Vector6 w0 = K_inv_ * s0_body + nominal_strain_;
    Vector6 w1 = K_inv_ * s1_body + nominal_strain_;

    // Assuming linear strain along the rod, compute total twist using Magnus expansion
    Matrix6 d_strain_d_w0, d_strain_d_w1;
    Vector6 strain_pred = get_strain_magnus(
        w0,
        w1,
        ds_,
        num_magnus_terms_,
        &d_strain_d_w0, 
        &d_strain_d_w1);
    
    // It should be equal to the actual strain velocity
    Vector6 error = strain_pred -  twist / ds_;

    if (H1) {
        *H1 = -1 / ds_ * d_twist_d_p0
            + d_strain_d_w0 * K_inv_ * d_s0body_d_p0;
    }

    if (H2) {
        *H2 = -1 / ds_ * d_twist_d_p1
            + d_strain_d_w1 * K_inv_ * d_s1body_d_p1;
    }

    if (H3) { *H3 = d_strain_d_w0 * K_inv_ * d_s0body_d_s0; }

    if (H4) { *H4 = d_strain_d_w1 * K_inv_ * d_s1body_d_s1; }

    return error;
}
