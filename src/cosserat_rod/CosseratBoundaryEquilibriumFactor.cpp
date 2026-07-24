#include "CosseratBoundaryEquilibriumFactor.h"

using namespace gtsam;

namespace {

KeyVector build_keys(Key stress_key, const std::optional<Key>& wrench_key)
{
    KeyVector keys = {stress_key};
    if (wrench_key) keys.push_back(*wrench_key);
    return keys;
}

}  // namespace

CosseratBoundaryEquilibriumFactor::CosseratBoundaryEquilibriumFactor(
    Key stress_key,
    std::optional<Key> wrench_key,
    const SharedNoiseModel& model,
    bool is_base)
:
    NoiseModelFactor(model, build_keys(stress_key, wrench_key)),
    is_base_(is_base)
{}

Vector CosseratBoundaryEquilibriumFactor::unwhitenedError(
    const Values& x, OptionalMatrixVecType H) const
{
    const Vector6 s = x.at<Vector6>(keys()[0]);

    // Free end: no wrench applied, so the stress here should be zero.
    if (keys().size() == 1) {
        if (H) (*H)[0] = Matrix6::Identity();
        return s;
    }

    // Otherwise, stress is constrained to be equal to the external wrench.
    //   base: stress flows out of the rod, so stress = -wrench;
    //   tip: stress flows into the applied wrench, stress = wrench.
    const Vector6 w = x.at<Vector6>(keys()[1]);
    double sign = is_base_ ? 1.0 : -1.0;

    if (H) {
        (*H)[0] = Matrix6::Identity();
        (*H)[1] = sign * Matrix6::Identity();
    }

    return s + sign * w;
}
