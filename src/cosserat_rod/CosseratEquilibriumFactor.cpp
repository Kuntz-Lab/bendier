#include "CosseratEquilibriumFactor.h"

#include "utils/WrenchTransforms.h"

using namespace gtsam;

namespace {

KeyVector build_keys(
    Key pose_0_key,
    Key pose_1_key,
    Key stress_0_key,
    Key stress_1_key,
    const std::optional<Key>& wrench_key)
{
    KeyVector keys = {pose_0_key, pose_1_key, stress_0_key, stress_1_key};
    if (wrench_key) keys.push_back(*wrench_key);
    return keys;
}

}  // namespace

CosseratEquilibriumFactor::CosseratEquilibriumFactor(
    Key pose_0_key,
    Key pose_1_key,
    Key stress_0_key,
    Key stress_1_key,
    std::optional<Key> wrench_key,
    const SharedNoiseModel& model)
:
    NoiseModelFactor(
        model, build_keys(pose_0_key, pose_1_key, stress_0_key, stress_1_key, wrench_key))
{}

Vector CosseratEquilibriumFactor::unwhitenedError(
    const Values& x, OptionalMatrixVecType H) const
{
    // Key layout:
    //   With wrench: [0]=pose_0, [1]=pose_1, [2]=stress_0, [3]=stress_1, [4]=wrench
    //   No wrench:   [0]=pose_0, [1]=pose_1, [2]=stress_0, [3]=stress_1
    const Pose3   p0 = x.at<Pose3>(keys()[0]);
    const Pose3   p1 = x.at<Pose3>(keys()[1]);
    const Vector6 s0 = x.at<Vector6>(keys()[2]);
    const Vector6 s1 = x.at<Vector6>(keys()[3]);
    const Vector6 w1 = (keys().size() == 5) ? x.at<Vector6>(keys()[4]) : Vector6::Zero();

    // Equilibrium equation, accounting for adjoint stress transport
    Matrix6 d_error_d_p0, d_error_d_p1, d_error_d_s0;
    Vector6 error = transform_wrench_translation(
        s0, p0, p1,
        d_error_d_s0, d_error_d_p0, d_error_d_p1) - w1 - s1;

    if (H) {
        (*H)[0] = d_error_d_p0;
        (*H)[1] = d_error_d_p1;
        (*H)[2] = d_error_d_s0;
        (*H)[3] = -Matrix6::Identity();
        if (keys().size() == 5) (*H)[4] = -Matrix6::Identity();
    }

    return error;
}
