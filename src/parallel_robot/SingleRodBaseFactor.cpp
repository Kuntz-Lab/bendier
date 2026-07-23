#include "SingleRodBaseFactor.h"
#include "utils/WrenchTransforms.h"

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
    // Twist of pose relative to the fixed mount pose
    Matrix6 d_xi_d_pose;
    Vector6 xi = pose_.localCoordinates(pose, std::nullopt, d_xi_d_pose);

    // Stress in the rod's body frame 
    Matrix6 d_stress_body_d_stress, d_stress_body_d_pose;
    Vector6 stress_body = spatial_to_body_wrench(stress, pose, d_stress_body_d_stress, d_stress_body_d_pose);

    // xi(2): rotation about the rod's mounting axis is omitted since the rod
    // spins freely about its axis so that DOF is unconstrained.
    // stress_body(2) = torsional stress (S_z in body frame) is constrained to
    // zero since the base joint is a bearing that cannot transmit axial torque.
    Vector6 error;
    error << xi(0), xi(1), xi(3), xi(4), xi(5), stress_body(2);

    if (H1) {
        Matrix6 H = Matrix6::Zero();

        H.row(0) = d_xi_d_pose.row(0);
        H.row(1) = d_xi_d_pose.row(1);
        H.row(2) = d_xi_d_pose.row(3);
        H.row(3) = d_xi_d_pose.row(4);
        H.row(4) = d_xi_d_pose.row(5);
        H.row(5) = d_stress_body_d_pose.row(2);

        *H1 = H;
    }

    if (H2) {
        Matrix6 H = Matrix6::Zero();
        H.row(5) = d_stress_body_d_stress.row(2);

        *H2 = H;
    }

    return error;
}
