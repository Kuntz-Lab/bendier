#include "WrenchTransforms.h"

using namespace gtsam;


Vector6 transform_wrench_adjoint(
    const Vector6& w0,
    const Pose3& p0,
    const Pose3& p,
    OptionalJacobian<6, 6> H_w0,
    OptionalJacobian<6, 6> H_p0,
    OptionalJacobian<6, 6> H_p) 
{   
    // Get pose p relative to p0
    Matrix6 d_delta_d_p0, d_delta_d_p;
    Pose3 delta = p0.between(p, d_delta_d_p0, d_delta_d_p);
    
    // Transforming wrench in delta's spatial frame to its body frame
    // i.e. from  p0  to p based on the above
    Matrix6 d_w_d_delta, d_w_d_w0;
    Vector6 w = delta.AdjointTranspose(w0, d_w_d_delta, d_w_d_w0);

    if (H_w0) { *H_w0 = d_w_d_w0; }

    if (H_p0) {
        *H_p0 = d_w_d_delta * d_delta_d_p0;
    }

    if (H_p) { *H_p = d_w_d_delta * d_delta_d_p; }
    
    return w;
}


Vector6 spatial_to_body_wrench(
    const Vector6& spatial, 
    const Pose3& pose, 
    OptionalJacobian<6, 6> H_spatial,
    OptionalJacobian<6, 6> H_pose)
{
    // Get rotation part of body pose
    Matrix36 d_rot_d_pose;
    Rot3 rot = pose.rotation(d_rot_d_pose);
    
    // Rotate components from world to body frame
    Matrix3 d_m_d_rot, d_m_d_m;
    Vector6 body;
    body.head<3>() = rot.unrotate(spatial.head<3>(), d_m_d_rot, d_m_d_m);
    
    Matrix3 d_f_d_rot, d_f_d_f;
    body.tail<3>() = rot.unrotate(spatial.tail<3>(), d_f_d_rot, d_f_d_f);

    if (H_pose) {
        Matrix63 d_body_d_rot = Matrix63::Zero();
        d_body_d_rot.block<3,3>(0,0) = d_m_d_rot;
        d_body_d_rot.block<3,3>(3,0) = d_f_d_rot;

        *H_pose = d_body_d_rot * d_rot_d_pose;
    }

    if (H_spatial) {
        H_spatial->setZero();
        H_spatial->block<3,3>(0,0) = d_m_d_m;
        H_spatial->block<3,3>(3,3) = d_f_d_f;
    }
    
    return body;
}


Vector6 body_to_spatial_wrench(
    const Vector6& body, 
    const Pose3& pose, 
    OptionalJacobian<6, 6> H_body,
    OptionalJacobian<6, 6> H_pose)
{
    // Same as above, except we use rotate instead of unrotate
    Matrix36 d_rot_d_pose;
    Rot3 rot = pose.rotation(d_rot_d_pose);
    
    Matrix3 d_m_d_rot, d_m_d_m;
    Vector6 spatial;
    spatial.head<3>() = rot.rotate(body.head<3>(), d_m_d_rot, d_m_d_m);
    
    Matrix3 d_f_d_rot, d_f_d_f;
    spatial.tail<3>() = rot.rotate(body.tail<3>(), d_f_d_rot, d_f_d_f);

    if (H_pose) {
        Matrix63 d_spatial_d_rot = Matrix63::Zero();
        d_spatial_d_rot.block<3,3>(0,0) = d_m_d_rot;
        d_spatial_d_rot.block<3,3>(3,0) = d_f_d_rot;

        *H_pose = d_spatial_d_rot * d_rot_d_pose;
    }

    if (H_body) {
        H_body->setZero();
        H_body->block<3,3>(0,0) = d_m_d_m;
        H_body->block<3,3>(3,3) = d_f_d_f;
    }
    
    return spatial;
}

