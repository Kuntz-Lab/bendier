#include "CosseratRodSolver.h"

#include <gtsam/nonlinear/PriorFactor.h>

#include "cosserat_rod/CosseratRodModel.h"
#include "utils/Gaussians.h"
#include "utils/MiscInline.h"

using namespace gtsam;

CosseratRodSolver::CosseratRodSolver(const CosseratRodSolverConfig& config)
:   
    SolverBase<CosseratRodModel>(config.base)
{
    SharedDiagonal strain_noise = get_noise_model_rot_pos(
        config.sigma_strain_rot, config.sigma_strain_pos);

    small_wrench_noise_ = get_noise_model_rot_pos(
        config.sigma_small_moment, config.sigma_small_force);

    base_pose_noise_ = get_noise_model_rot_pos(
        config.sigma_base_pose_rot, config.sigma_base_pose_pos);

    model_ = std::make_unique<CosseratRodModel>(CosseratRodModelConfig{
        .num_nodes = config.num_nodes,
        .K_inv = config.K_inv,
        .strain_noise = strain_noise,
        .stress_noise = small_wrench_noise_,
        .num_magnus_terms = config.num_magnus_terms,
        .rod_length = config.rod_length,
        .wrench_node_indices = {0, config.num_nodes - 1},  // Only base and tip have wrenches for this solver
    });
}

Solution<CosseratRodModel::ModelMarginals> CosseratRodSolver::solve(
    const std::optional<Vector6Gaussian>& tip_wrench,
    const std::optional<Pose3Gaussian>&   tip_pose,
    const std::optional<Vector6>&         nominal_strain)
{
    if (nominal_strain)
        model_->set_nominal_strain(*nominal_strain);

    NonlinearFactorGraph priors;
    priors.add(PriorFactor<Pose3>(
        model_->get_pose_key(0), Pose3::Identity(), base_pose_noise_));

    if (tip_wrench)
        priors.add(PriorFactor<Vector6>(
            model_->get_wrench_key(-1),
            tip_wrench->mean,
            noiseModel::Gaussian::Covariance(tip_wrench->cov)));

    if (tip_pose)
        priors.add(PriorFactor<Pose3>(
            model_->get_pose_key(-1),
            Pose3(tip_pose->mean),
            noiseModel::Gaussian::Covariance(tip_pose->cov)));

    return run_solve(std::move(priors));
}
