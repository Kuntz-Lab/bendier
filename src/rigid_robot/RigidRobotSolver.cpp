#include "RigidRobotSolver.h"
#include "RigidJointTorqueFactor.h"

#include <gtsam/nonlinear/PriorFactor.h>
#include <stdexcept>

#include "utils/MiscInline.h"

using namespace gtsam;

RigidRobotSolver::RigidRobotSolver(const RigidRobotSolverConfig& config)
:
    SolverBase<RigidRobotModel>(config.base)
{
    SharedDiagonal chain_noise = get_noise_model_rot_pos(
        config.sigma_chain_rot, config.sigma_chain_pos);

    model_ = std::make_unique<RigidRobotModel>(RigidRobotModelConfig{
        .joints = config.joints,
        .base_pose_calibration = config.base_pose_calibration,
        .tip_offset_calibration = config.tip_offset_calibration,
        .chain_noise = chain_noise,
        .enable_wrench_sensing = config.enable_wrench_sensing,
    });
}

Solution<RigidRobotModel::ModelMarginals> RigidRobotSolver::solve(
    const VectorXGaussian& joint_prior,
    const std::optional<Vector6Gaussian>& tip_wrench_prior,
    const std::optional<VectorXGaussian>& joint_torque_meas,
    const std::optional<Pose3Gaussian>& tip_pose_prior)
{
    // TODO this is likely where we would check if either tip_wrench_prior or joint_torque_meas is given if enable_wrench_sensing is true, don't need to expose a setter etc. for that.
    if (joint_prior.mean.size() != model_->get_num_joints())
        throw std::invalid_argument("RigidRobotSolver: joint_prior size must match num_joints");

    if ((tip_wrench_prior || joint_torque_meas) && !model_->has_wrench_sensing())
        throw std::invalid_argument(
            "RigidRobotSolver: tip_wrench_prior/joint_torque_meas given but "
            "enable_wrench_sensing is false");

    if (model_->has_wrench_sensing() && !tip_wrench_prior && !joint_torque_meas)
        throw std::invalid_argument(
            "RigidRobotSolver: enable_wrench_sensing is true but neither "
            "tip_wrench_prior nor joint_torque_meas was given -- the tip "
            "wrench variable would be completely unconstrained");

    // Fir first solve, we have to get the initial default values from the model 
    if (warm_start_.empty())
        warm_start_ = model_->get_initial_values(std::nullopt, joint_prior.mean);

    // Build prior factors from the user inputs
    NonlinearFactorGraph priors;
    
    priors.add(PriorFactor<Vector>(
        model_->get_joint_vector_key(),
        joint_prior.mean,
        noiseModel::Gaussian::Covariance(joint_prior.cov)));

    if (tip_wrench_prior)
        priors.add(PriorFactor<Vector6>(
            model_->get_tip_wrench_key(),
            tip_wrench_prior->mean,
            noiseModel::Gaussian::Covariance(tip_wrench_prior->cov)));

    if (tip_pose_prior)
        priors.add(PriorFactor<Pose3>(
            model_->get_tip_pose_key(),
            Pose3(tip_pose_prior->mean),
            noiseModel::Gaussian::Covariance(tip_pose_prior->cov)));

    if (joint_torque_meas) {
        // TODO: this seems a little hacky
        int num_joints = model_->get_num_joints();
        if (joint_torque_meas->mean.size() != num_joints)
            throw std::invalid_argument(
                "RigidRobotSolver: joint_torque_meas size must match num_joints");

        Key pose_tip_key = model_->get_tip_pose_key();
        Key wrench_key = model_->get_tip_wrench_key();

        for (int i = 0; i < num_joints; ++i) {
            double sigma = std::sqrt(joint_torque_meas->cov(i, i));

            priors.add(RigidJointTorqueFactor(
                pose_tip_key,
                model_->get_pose_key(i + 1),
                wrench_key,
                model_->get_joint_axis(i),
                model_->get_joint_type(i),
                joint_torque_meas->mean(i),
                noiseModel::Isotropic::Sigma(1, sigma)));
        }
    }

    return run_solve(std::move(priors));
}
