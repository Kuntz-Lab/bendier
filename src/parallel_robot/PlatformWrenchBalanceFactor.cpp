#include "PlatformWrenchBalanceFactor.h"

#include "utils/WrenchTransforms.h"

using namespace gtsam;

PlatformWrenchBalanceFactor::PlatformWrenchBalanceFactor(
    Key stress_key_0, Key pose_key_0,
    Key stress_key_1, Key pose_key_1,
    Key stress_key_2, Key pose_key_2,
    Key stress_key_3, Key pose_key_3,
    Key stress_key_4, Key pose_key_4,
    Key stress_key_5, Key pose_key_5,
    Key platform_wrench_key, Key platform_pose_key,
    const SharedNoiseModel& model)
:
    PlatformWrenchBase(model, 
        stress_key_0, pose_key_0,
        stress_key_1, pose_key_1,
        stress_key_2, pose_key_2,
        stress_key_3, pose_key_3,
        stress_key_4, pose_key_4,
        stress_key_5, pose_key_5,
        platform_wrench_key, platform_pose_key) 
{}

Vector PlatformWrenchBalanceFactor::evaluateError(
    const Vector6& stress_0, const Pose3& pose_0,
    const Vector6& stress_1, const Pose3& pose_1,
    const Vector6& stress_2, const Pose3& pose_2,
    const Vector6& stress_3, const Pose3& pose_3,
    const Vector6& stress_4, const Pose3& pose_4,
    const Vector6& stress_5, const Pose3& pose_5,
    const Vector6& platform_wrench, const Pose3& platform_pose,
    OptionalMatrixType H1, OptionalMatrixType H2,
    OptionalMatrixType H3, OptionalMatrixType H4, 
    OptionalMatrixType H5, OptionalMatrixType H6,
    OptionalMatrixType H7, OptionalMatrixType H8,
    OptionalMatrixType H9, OptionalMatrixType H10, 
    OptionalMatrixType H11, OptionalMatrixType H12,
    OptionalMatrixType H13, OptionalMatrixType H14) const 
{
    Matrix6 d_s0_p_d_pp, d_s1_p_d_pp, d_s2_p_d_pp, d_s3_p_d_pp, d_s4_p_d_pp, d_s5_p_d_pp;
    Vector6 s0_p = transform_wrench_translation(stress_0, pose_0, platform_pose,
        H1 ? H1 : nullptr, H2 ? H2 : nullptr, d_s0_p_d_pp);

    Vector6 s1_p = transform_wrench_translation(stress_1, pose_1, platform_pose,
        H3 ? H3 : nullptr, H4 ? H4 : nullptr, d_s1_p_d_pp);

    Vector6 s2_p = transform_wrench_translation(stress_2, pose_2, platform_pose,
        H5 ? H5 : nullptr, H6 ? H6 : nullptr, d_s2_p_d_pp);

    Vector6 s3_p = transform_wrench_translation(stress_3, pose_3, platform_pose,
        H7 ? H7 : nullptr, H8 ? H8 : nullptr, d_s3_p_d_pp);

    Vector6 s4_p = transform_wrench_translation(stress_4, pose_4, platform_pose,
        H9 ? H9 : nullptr, H10 ? H10 : nullptr, d_s4_p_d_pp);

    Vector6 s5_p = transform_wrench_translation(stress_5, pose_5, platform_pose,
        H11 ? H11 : nullptr, H12 ? H12 : nullptr, d_s5_p_d_pp);

    Vector6 stress_error = s0_p + s1_p + s2_p + s3_p + s4_p + s5_p - platform_wrench;

    if (H13) { *H13 = -Matrix6::Identity(); }

    if (H14) {
        *H14 = d_s0_p_d_pp + d_s1_p_d_pp + d_s2_p_d_pp + d_s3_p_d_pp + d_s4_p_d_pp + d_s5_p_d_pp;
    }

    return stress_error;
}
