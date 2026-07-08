#include "WrenchTransforms.h"

using namespace gtsam;

Vector6 transform_wrench_translation(
    const Vector6& w0,
    const Pose3& p0,
    const Pose3& p,
    OptionalJacobian<6, 6> H_w0,
    OptionalJacobian<6, 6> H_p0,
    OptionalJacobian<6, 6> H_p)
{
    Matrix36 d_t0_d_p0, d_t1_d_p;
    Vector3 t0 = p0.translation(d_t0_d_p0);
    Vector3 t1 = p.translation(d_t1_d_p);

    Vector3 m0 = w0.head<3>();
    Vector3 n0 = w0.tail<3>();

    Vector3 d = t1 - t0;
    Matrix3 d_skew = skewSymmetric(d);
    Matrix3 n0_skew = skewSymmetric(n0);

    Vector6 w;
    w.head<3>() = m0 - d_skew * n0;
    w.tail<3>() = n0;

    if (H_w0) {
        H_w0->setZero();
        H_w0->block<3,3>(0,0) = Matrix3::Identity();
        H_w0->block<3,3>(0,3) = -d_skew;
        H_w0->block<3,3>(3,3) = Matrix3::Identity();
    }

    if (H_p0) {
        H_p0->setZero();
        H_p0->block<3,6>(0,0) = -n0_skew * d_t0_d_p0;
    }

    if (H_p) {
        H_p->setZero();
        H_p->block<3,6>(0,0) = n0_skew * d_t1_d_p;
    }

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
