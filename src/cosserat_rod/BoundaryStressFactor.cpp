#include "BoundaryStressFactor.h"

using namespace gtsam;

BoundaryStressFactor::BoundaryStressFactor(
    Key stress_key,
    Key wrench_key,
    const SharedNoiseModel& model,
    bool is_base)
:
    is_base_(is_base),
    BoundaryStressBase(model, stress_key, wrench_key)
{}

Vector BoundaryStressFactor::evaluateError(
    const Vector6& stress,
    const Vector6& wrench,
    OptionalMatrixType H1,
    OptionalMatrixType H2) const
{
    // At the base, the stress is negative wrench, since it flows out of the rod
    double sign = is_base_ ? 1.0 : -1.0;

    Vector6 stress_error = stress + sign * wrench;

    if (H1) { *H1 = Matrix6::Identity(); }

    if (H2) { *H2 = sign * Matrix6::Identity(); }

    return stress_error;
}
