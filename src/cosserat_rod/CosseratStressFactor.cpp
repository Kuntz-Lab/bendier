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
    NoiseModelFactor(
        model, 
        KeyVector{pose_0_key, pose_1_key, stress_0_key, stress_1_key, wrench_key}) 
{}

CosseratStressFactor::CosseratStressFactor(
    Key pose_0_key, 
    Key pose_1_key,
    Key stress_0_key, 
    Key stress_1_key,
    const SharedNoiseModel& model)
:
    NoiseModelFactor(
        model, 
        KeyVector{pose_0_key, pose_1_key, stress_0_key, stress_1_key}) 
{}

Vector CosseratStressFactor::unwhitenedError(
    const Values& x, OptionalMatrixVecType H) const
{
    // Key layout:
    //   Interior: [0]=pose_0, [1]=pose_1, [2]=stress_0, [3]=stress_1, [4]=wrench_1
    //   Tip:      [0]=pose_0, [1]=pose_1, [2]=stress_0, [3]=stress_1
    const Pose3   p0 = x.at<Pose3>(keys()[0]);
    const Pose3   p1 = x.at<Pose3>(keys()[1]);
    const Vector6 s0 = x.at<Vector6>(keys()[2]);
    const Vector6 s1 = x.at<Vector6>(keys()[3]);
    const Vector6 w1 = (keys().size() == 5) ? x.at<Vector6>(keys()[4]) : Vector6::Zero();

    Matrix6 d_s1_pred_d_p0, d_s1_pred_d_p1, d_s1_pred_d_s0;
    Vector6 s1_pred = transform_wrench_translation(
        s0, p0, p1,
        d_s1_pred_d_s0, d_s1_pred_d_p0, d_s1_pred_d_p1) - w1;

    if (H) {
        (*H)[0] = d_s1_pred_d_p0;
        (*H)[1] = d_s1_pred_d_p1;
        (*H)[2] = d_s1_pred_d_s0;
        (*H)[3] = -Matrix6::Identity();
        if (keys().size() == 5) (*H)[4] = -Matrix6::Identity();
    }

    return s1_pred - s1;
}
