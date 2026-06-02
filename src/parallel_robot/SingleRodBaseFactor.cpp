#include "SingleRodBaseFactor.h"

using namespace gtsam;


SingleRodBaseFactor::SingleRodBaseFactor(
    Key pose_key,
    Key stress_key,
    const Pose3& pose,
    const SharedNoiseModel& model)
:
    NoiseModelFactorN<gtsam::Pose3, gtsam::Vector6>(model, pose_key, stress_key),
    pose_(pose) 
{}


Vector SingleRodBaseFactor::evaluateError(
    const Pose3& pose,
    const Vector6& stress,
    OptionalMatrixType H1,
    OptionalMatrixType H2) const 
{
    Matrix6 d_delta_d_pose;
    Pose3 delta = pose_.between(pose, std::nullopt, d_delta_d_pose);

    Matrix6 d_xi_d_delta;
    Vector6 xi = Pose3::Logmap(delta, d_xi_d_delta);

    // Remove the z rotation error, it is free to move
    Vector6 error;
    error << xi(0), xi(1), xi(3), xi(4), xi(5), stress(2);

    if (H1) {
        Matrix6 J = d_xi_d_delta * d_delta_d_pose;

        Matrix6 H = Matrix6::Zero();

        H.row(0) = J.row(0);
        H.row(1) = J.row(1);
        H.row(2) = J.row(3);
        H.row(3) = J.row(4);
        H.row(4) = J.row(5);
        // row 5 stays zero

        *H1 = H;
    }

    if (H2) {
        Matrix6 H = Matrix6::Zero();
        H(5,2) = 1.0;   // derivative of stress(2)

        *H2 = H;
    }

    return error;
}