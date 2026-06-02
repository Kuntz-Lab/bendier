#include "ParallelRobotSolver.h"

#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>

#include "parallel_robot/ParallelRobotModel.h"
#include "measurement/ActuationForceMeasFactor.h"
#include "utils/MiscInline.h"
#include "utils/SolverBase.h"

using namespace gtsam;


ParallelRobotSolver::ParallelRobotSolver(const ParallelRobotSolverConfig& config) 
:
    SolverBase(config.base)
{
    SharedDiagonal strain_noise = get_noise_model_rot_pos(
        config.sigma_strain_rot, config.sigma_strain_pos); 
    
    small_wrench_noise_ = get_noise_model_rot_pos(
        config.sigma_small_moment, config.sigma_small_force); 
    
    robot_ = std::make_unique<ParallelRobot>(
        config.nodes_per_rod, 
        config.K_inv,
        strain_noise,
        small_wrench_noise_,
        config.base_end_poses,
        config.tip_end_poses,
        config.sigma_end_pose_pos,
        config.sigma_end_pose_rot);

    get_initial_values();
}


void ParallelRobotSolver::get_initial_values() {
    values_ = robot_->get_initial_values();
}


void ParallelRobotSolver::build_graph() {
    graph_ = robot_->build_graph(rod_lengths_, sigma_rod_lengths_, wrench_);

    // If we have actuation force measurements, use them
    if (f_meas_) {
        auto noise = noiseModel::Isotropic::Sigma(1, f_meas_->sigma);
        for (int i = 0; i < NUM_RODS; i++) {
            graph_.add(ActuationForceMeasFactor(
                robot_->rods_[i]->get_wrench_key(0),
                f_meas_->meas[i],
                noise));
        }
    }
}


void ParallelRobotSolver::extract_solution() {
    extracted_ = robot_->get_marginals(values_, marginals_);
    extracted_.rod_lengths_jacobian = robot_->get_rod_lengths_jacobian(marginals_);
    extracted_.tip_wrench_jacobian = robot_->get_tip_wrench_jacobian(marginals_);
}

    
Solution<ParallelRobotMarginals> ParallelRobotSolver::solve(
    const std::array<double, NUM_RODS>& rod_lengths, 
    double sigma_rod_lengths,
    const Vector6Gaussian& wrench,
    const std::optional<ActuationForceMeas>& f_meas) 
{
    rod_lengths_ = rod_lengths;
    sigma_rod_lengths_ = sigma_rod_lengths;
    wrench_ = wrench;
    f_meas_ = f_meas;

    Solution<ParallelRobotMarginals> solution;
    solution.meta = optimize();
    solution.marginals = extracted_;

    return solution;
}