#include "ParallelRobotSolver.h"

#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/PriorFactor.h>

#include "parallel_robot/ParallelRobotModel.h"
#include "measurement/ActuationForceMeasFactor.h"
#include "utils/MiscInline.h"

using namespace gtsam;

ParallelRobotSolver::ParallelRobotSolver(const ParallelRobotSolverConfig& config)
:   SolverBase<ParallelRobotModel>(config.base)
{
    SharedDiagonal strain_noise = get_noise_model_rot_pos(
        config.sigma_strain_rot, config.sigma_strain_pos);

    SharedDiagonal small_wrench_noise = get_noise_model_rot_pos(
        config.sigma_small_moment, config.sigma_small_force);

    model_ = std::make_unique<ParallelRobotModel>(
        config.nodes_per_rod,
        config.K_inv,
        strain_noise,
        small_wrench_noise,
        config.base_end_poses,
        config.tip_end_poses,
        config.sigma_end_pose_pos,
        config.sigma_end_pose_rot,
        /* sigma_rod_lengths = */ 1e-3); // We overwrite this on each solve, initial value doesn't matter.
}

Solution<ParallelRobotModel::ModelMarginals> ParallelRobotSolver::solve(
    const Vector&                        rod_lengths,
    double                              sigma_rod_lengths,
    const Vector6Gaussian&              wrench,
    const std::optional<ActuationForceMeas>& f_meas)
{
    model_->set_rod_lengths(rod_lengths);
    model_->set_sigma_rod_lengths(sigma_rod_lengths);

    NonlinearFactorGraph priors;
    priors.add(PriorFactor<Vector6>(
        model_->platform_wrench_key(),
        wrench.mean,
        noiseModel::Gaussian::Covariance(wrench.cov)));

    if (f_meas) {
        auto noise = noiseModel::Isotropic::Sigma(1, f_meas->sigma);
        for (int i = 0; i < model_->get_num_rods(); i++)
            // We are using the world frame wrench here, but in a realistic scenario, the measured wrench would be in the rod frame.
            // We could add a wrench rotation into the measurement model inside the factor if we wanted to do that. Fine for now.
            priors.add(ActuationForceMeasFactor(
                model_->get_rod_wrench_key(i, 0), f_meas->meas[i], noise));
    }

    return run_solve(std::move(priors));
}
