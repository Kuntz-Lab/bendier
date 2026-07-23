#include "TendonRobotSolver.h"

#include <gtsam/nonlinear/PriorFactor.h>

#include "utils/Gaussians.h"
#include "utils/MiscInline.h"
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/slam/PoseTranslationPrior.h>

using namespace gtsam;

TendonRobotSolver::TendonRobotSolver(const TendonRobotSolverConfig& config)
:   SolverBase<TendonRobotModel>(config.base)
{
    SharedDiagonal strain_noise = get_noise_model_rot_pos(
        config.sigma_strain_rot, config.sigma_strain_pos);

    small_wrench_noise_ = get_noise_model_rot_pos(
        config.sigma_small_moment, config.sigma_small_force);

    Rot3  base_rot       = Rot3::Rx(-M_PI / 2).compose(Rot3::Rz(M_PI));
    Pose3 base_pose_mean = Pose3(base_rot, Point3::Zero());
    SharedDiagonal base_pose_noise = get_noise_model_rot_pos(
        config.sigma_base_pose_rot, config.sigma_base_pose_pos);

    SharedDiagonal displacement_constraint_noise = noiseModel::Isotropic::Sigma(
        static_cast<int>(config.tendon_input.functions.size()), config.sigma_displacement_constraint);

    int num_nodes = TendonRobotNumNodes(config.num_discs, config.num_between_nodes);

    // Only base and tip are true external wrench nodes for this solver
    model_ = std::make_unique<TendonRobotModel>(
        config.rod_length,
        config.num_discs,
        config.num_between_nodes,
        config.tendon_input,
        config.K_inv,
        strain_noise,
        small_wrench_noise_,
        displacement_constraint_noise,
        config.axial_stiffness,
        base_pose_mean,
        base_pose_noise,
        std::vector<int>{0, num_nodes - 1});
}

Solution<TendonRobotModel::ModelMarginals> TendonRobotSolver::solve(
    const VectorXGaussian&                tensions,
    const std::optional<Vector6Gaussian>& tip_wrench,
    const std::optional<Vector3Gaussian>& tip_position_meas,
    const std::optional<VectorXGaussian>& displacement_meas)
{
    NonlinearFactorGraph priors;
    priors.add(PriorFactor<Vector>(
        model_->get_tensions_key(),
        tensions.mean,
        noiseModel::Gaussian::Covariance(tensions.cov)));

    if (displacement_meas)
        priors.add(PriorFactor<Vector>(
            model_->get_displacements_key(),
            displacement_meas->mean,
            noiseModel::Gaussian::Covariance(displacement_meas->cov)));

    int num_nodes = model_->get_num_nodes();

    Vector6          tip_wrench_mean  = Vector6::Zero();
    SharedNoiseModel tip_wrench_noise = small_wrench_noise_;
    if (tip_wrench) {
        tip_wrench_mean  = tip_wrench->mean;
        tip_wrench_noise = noiseModel::Gaussian::Covariance(tip_wrench->cov);
    }
    priors.add(PriorFactor<Vector6>(
        *model_->get_external_wrench_key(num_nodes - 1), tip_wrench_mean, tip_wrench_noise));

    if (tip_position_meas)
        priors.add(PoseTranslationPrior<Pose3>(
            model_->get_pose_key(-1),
            tip_position_meas->mean,
            noiseModel::Gaussian::Covariance(tip_position_meas->cov)));

    return run_solve(std::move(priors));
}
