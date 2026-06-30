#include "ActuationForceMeasFactor.h"

using namespace gtsam;

ActuationForceMeasFactor::ActuationForceMeasFactor(
    Key wrench_key,
    double f_z_meas,
    const SharedNoiseModel& model) 
:
    f_z_meas_(f_z_meas),
    NoiseModelFactor1<Vector6>(model, wrench_key) 
{}

Vector ActuationForceMeasFactor::evaluateError(
    const Vector6& wrench, 
    OptionalMatrixType H1) const
{
    Vector1 e;
    e(0) = wrench[5] - f_z_meas_;

    if (H1) {
        Matrix16 H = Matrix16::Zero();
        H(0,5) = 1.0;
        *H1 = H;
    }

    return e;
}
