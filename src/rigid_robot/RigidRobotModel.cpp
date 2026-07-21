#include "RigidRobotModel.h"
#include "RigidJointTorqueFactor.h"
#include "utils/ModelConcept.h"

#include <gtsam/nonlinear/PriorFactor.h>
#include <stdexcept>

using namespace gtsam;

static_assert(BendierModel<RigidRobotModel>);

RigidRobotModel::RigidRobotModel(const RigidRobotModelConfig& config)
:
    id_(next_id_++),
    num_joints_(static_cast<int>(config.joints.size())),
    num_links_(num_joints_ + 1),
    joint_specs_(config.joints),
    base_pose_calibration_(config.base_pose_calibration),
    chain_noise_(config.chain_noise),
    enable_wrench_sensing_(config.enable_wrench_sensing)
{
    pose_keys_.reserve(num_links_);
    for (int i = 0; i < num_links_; ++i)
        pose_keys_.push_back(Symbol('T', 1000 * id_ + i));

    offset_keys_.reserve(num_joints_);
    for (int i = 0; i < num_joints_; ++i)
        offset_keys_.push_back(Symbol('O', 1000 * id_ + i));
}

int RigidRobotModel::clamp_link_idx(int link_idx) const {
    if (link_idx == -1)
        return num_links_ - 1;

    if (link_idx < 0 || link_idx >= num_links_)
        throw std::out_of_range("RigidRobotModel: invalid link_idx");

    return link_idx;
}

int RigidRobotModel::clamp_joint_idx(int joint_idx) const {
    if (joint_idx == -1)
        return num_joints_ - 1;

    if (joint_idx < 0 || joint_idx >= num_joints_)
        throw std::out_of_range("RigidRobotModel: invalid joint_idx");

    return joint_idx;
}

Key RigidRobotModel::get_pose_key(int link_idx) const { return pose_keys_[clamp_link_idx(link_idx)]; }
Key RigidRobotModel::get_offset_key(int joint_idx) const { return offset_keys_[clamp_joint_idx(joint_idx)]; }
Key RigidRobotModel::get_joint_vector_key() const { return Symbol('Q', id_); }
Key RigidRobotModel::get_tip_wrench_key() const { return Symbol('W', id_); }

const Vector3& RigidRobotModel::get_joint_axis(int joint_idx) const {
    return joint_specs_[clamp_joint_idx(joint_idx)].axis;
}

JointType RigidRobotModel::get_joint_type(int joint_idx) const {
    return joint_specs_[clamp_joint_idx(joint_idx)].type;
}

const std::vector<Key>& RigidRobotModel::get_pose_keys() const { return pose_keys_; }

static Vector6 joint_twist(const RigidJointSpec& spec, double q) {
    Vector6 xi = Vector6::Zero();
    if (spec.type == JointType::Revolute) xi.head<3>() = spec.axis * q;
    else                                  xi.tail<3>() = spec.axis * q;
    return xi;
}

Values RigidRobotModel::get_initial_values(
    const std::optional<Pose3>& base_pose_init,
    const Vector& nominal_joint_values) const
{
    Values values;

    Vector q = (nominal_joint_values.size() == num_joints_)
        ? nominal_joint_values : Vector::Zero(num_joints_);

    values.insert(get_joint_vector_key(), q);

    if (enable_wrench_sensing_)
        values.insert(get_tip_wrench_key(), Vector6(Vector6::Zero()));

    Pose3 pose = base_pose_init ? *base_pose_init : Pose3(base_pose_calibration_.mean);
    values.insert(pose_keys_[0], pose);

    for (int i = 0; i < num_joints_; ++i) {
        Pose3 offset = Pose3(joint_specs_[i].offset_calibration.mean);
        values.insert(offset_keys_[i], offset);

        pose = pose * offset * Pose3::Expmap(joint_twist(joint_specs_[i], q[i]));
        values.insert(pose_keys_[i + 1], pose);
    }

    return values;
}

NonlinearFactorGraph RigidRobotModel::build_graph() const
{
    NonlinearFactorGraph graph;
    Key joint_key = get_joint_vector_key();

    graph.add(PriorFactor<Pose3>(
        pose_keys_[0],
        Pose3(base_pose_calibration_.mean),
        noiseModel::Gaussian::Covariance(base_pose_calibration_.cov)));

    for (int i = 0; i < num_joints_; ++i) {
        graph.add(RigidJointFactor(
            pose_keys_[i],
            pose_keys_[i + 1],
            offset_keys_[i],
            joint_key,
            i,
            joint_specs_[i].axis,
            joint_specs_[i].type,
            chain_noise_));

        graph.add(PriorFactor<Pose3>(
            offset_keys_[i],
            Pose3(joint_specs_[i].offset_calibration.mean),
            noiseModel::Gaussian::Covariance(joint_specs_[i].offset_calibration.cov)));
    }

    // No internal physics factor needed for the tip wrench itself: with no
    // distributed load along a rigid link, it's the same wrench at every
    // joint, just transported (see RigidJointTorqueFactor) -- there's
    // nothing to balance here the way CosseratStressFactor balances an
    // elastic rod's internal stress node-to-node. The solver is entirely
    // responsible for constraining it (tip-wrench prior and/or per-joint
    // torque measurements).

    return graph;
}

RigidRobotMarginals RigidRobotModel::get_marginals(
    const Values& values,
    const Marginals& marginals) const
{
    RigidRobotMarginals out;
    out.links.resize(num_links_);
    out.offsets.resize(num_joints_);

    for (int i = 0; i < num_links_; ++i) {
        out.links[i].pose.mean = values.at<Pose3>(pose_keys_[i]).matrix();
        out.links[i].pose.cov  = marginals.marginalCovariance(pose_keys_[i]);
    }

    for (int i = 0; i < num_joints_; ++i) {
        out.offsets[i].mean = values.at<Pose3>(offset_keys_[i]).matrix();
        out.offsets[i].cov  = marginals.marginalCovariance(offset_keys_[i]);
    }

    Key joint_key = get_joint_vector_key();
    out.joints.mean = values.at<Vector>(joint_key);
    out.joints.cov  = marginals.marginalCovariance(joint_key);

    if (enable_wrench_sensing_) {
        Key wrench_key = get_tip_wrench_key();
        Vector6 tip_wrench_mean = values.at<Vector6>(wrench_key);
        Matrix6 tip_wrench_cov = marginals.marginalCovariance(wrench_key);

        Vector6Gaussian tip_wrench;
        tip_wrench.mean = tip_wrench_mean;
        tip_wrench.cov = tip_wrench_cov;
        out.tip_wrench = tip_wrench;

        // Per-joint generalized force, projected from the tip wrench at the
        // solved poses. Variance only propagates the tip wrench's own
        // uncertainty through this projection's Jacobian (poses are held at
        // their MAP estimate) -- reuses RigidJointTorqueFactor's own
        // evaluateError/Jacobian rather than re-deriving the same math.
        VectorXGaussian joint_torques;
        joint_torques.mean = Vector::Zero(num_joints_);
        joint_torques.cov = Matrix::Zero(num_joints_, num_joints_);

        Pose3 pose_tip = values.at<Pose3>(pose_keys_.back());
        for (int i = 0; i < num_joints_; ++i) {
            Pose3 pose_child = values.at<Pose3>(pose_keys_[i + 1]);

            RigidJointTorqueFactor factor(
                pose_keys_.back(), pose_keys_[i + 1], wrench_key,
                joint_specs_[i].axis, joint_specs_[i].type,
                /*torque_meas=*/0.0, noiseModel::Unit::Create(1));

            Matrix H3;
            Vector error = factor.evaluateError(
                pose_tip, pose_child, tip_wrench_mean, nullptr, nullptr, &H3);

            joint_torques.mean(i) = error(0);  // torque_meas is 0, so error == predicted torque
            joint_torques.cov(i, i) = (H3 * tip_wrench_cov * H3.transpose())(0, 0);
        }

        out.joint_torques = joint_torques;
    }

    return out;
}
