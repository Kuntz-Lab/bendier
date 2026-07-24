#include "PlatformWrenchBalanceFactor.h"

#include <stdexcept>

#include "utils/WrenchTransforms.h"

using namespace gtsam;

namespace {

KeyVector make_keys(
    const KeyVector& tip_wrench_keys,
    const KeyVector& pose_keys,
    Key platform_wrench_key,
    Key platform_pose_key)
{
    KeyVector keys;
    keys.reserve(2 * tip_wrench_keys.size() + 2);
    for (size_t i = 0; i < tip_wrench_keys.size(); ++i) {
        keys.push_back(tip_wrench_keys[i]);
        keys.push_back(pose_keys[i]);
    }
    keys.push_back(platform_wrench_key);
    keys.push_back(platform_pose_key);
    return keys;
}

}  // namespace

PlatformWrenchBalanceFactor::PlatformWrenchBalanceFactor(
    const KeyVector& tip_wrench_keys,
    const KeyVector& pose_keys,
    Key platform_wrench_key,
    Key platform_pose_key,
    const SharedNoiseModel& model)
:
    NoiseModelFactor(model, make_keys(tip_wrench_keys, pose_keys, platform_wrench_key, platform_pose_key)),
    num_rods_(tip_wrench_keys.size())
{
    if (tip_wrench_keys.size() != pose_keys.size())
        throw std::invalid_argument(
            "PlatformWrenchBalanceFactor: tip_wrench_keys and pose_keys must be the same size");
}

Vector PlatformWrenchBalanceFactor::unwhitenedError(
    const Values& x, OptionalMatrixVecType H) const
{
    // Key layout: [tip_wrench_0, pose_0, ..., tip_wrench_{N-1}, pose_{N-1}, platform_wrench, platform_pose]
    const Key wrench_key = keys()[2 * num_rods_];
    const Key pose_key   = keys()[2 * num_rods_ + 1];

    const Vector6 platform_wrench = x.at<Vector6>(wrench_key);
    const Pose3   platform_pose   = x.at<Pose3>(pose_key);

    // Sum of all rod tip wrenches, transported to the platform's position
    Vector6 sum = Vector6::Zero();
    Matrix6 d_sum_d_platform_pose = Matrix6::Zero();

    for (size_t i = 0; i < num_rods_; ++i) {
        const Vector6 tip_wrench_i = x.at<Vector6>(keys()[2 * i]);
        const Pose3   pose_i       = x.at<Pose3>(keys()[2 * i + 1]);

        Matrix6 d_wi_p_d_tip_wrench_i, d_wi_p_d_pose_i, d_wi_p_d_platform_pose;
        Vector6 wi_p = transform_wrench_translation(
            tip_wrench_i, pose_i, platform_pose,
            d_wi_p_d_tip_wrench_i, d_wi_p_d_pose_i, d_wi_p_d_platform_pose);

        sum += wi_p;
        d_sum_d_platform_pose += d_wi_p_d_platform_pose;

        if (H) {
            (*H)[2 * i]     = d_wi_p_d_tip_wrench_i;
            (*H)[2 * i + 1] = d_wi_p_d_pose_i;
        }
    }

    if (H) {
        (*H)[2 * num_rods_]     = -Matrix6::Identity();
        (*H)[2 * num_rods_ + 1] = d_sum_d_platform_pose;
    }

    return sum - platform_wrench;
}
