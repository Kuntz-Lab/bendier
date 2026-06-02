#include "PositionPriorFactor.h"

using namespace gtsam;


PositionPriorFactor::PositionPriorFactor(
    Key pose_key,
    Vector3 position_meas,
    const SharedNoiseModel& model)
: 
    NoiseModelFactorN<gtsam::Pose3>(model, pose_key), position_meas_(position_meas) {}


Vector PositionPriorFactor::evaluateError(const Pose3& pose, OptionalMatrixType H1) const {  
    Matrix36 d_position_d_pose;
    Vector3 error = pose.translation(d_position_d_pose) - position_meas_;

    if (H1) { *H1 = d_position_d_pose; }

    return error;
}