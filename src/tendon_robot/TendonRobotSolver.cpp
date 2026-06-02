#include "TendonRobotSolver.h"

#include "measurement/PositionPriorFactor.h"
#include "utils/Gaussians.h"
#include "utils/MiscInline.h"
#include "utils/SolverBase.h"
#include <gtsam/linear/NoiseModel.h>

using namespace gtsam;


TendonRobotSolver::TendonRobotSolver(const TendonRobotSolverConfig& config) 
:
    SolverBase(config.base)
{
    SharedDiagonal strain_noise = get_noise_model_rot_pos(
        config.sigma_strain_rot, config.sigma_strain_pos); 
    
    small_wrench_noise_ = get_noise_model_rot_pos(
        config.sigma_stress_moment, config.sigma_stress_force); 
    
    Rot3 base_rot = Rot3::Rx(-M_PI / 2).compose(Rot3::Rz(M_PI));
    Pose3 base_pose_mean = Pose3(base_rot, Point3::Zero());
    SharedDiagonal base_pose_noise = get_noise_model_rot_pos(
        config.sigma_base_rot, config.sigma_base_pos);
    
    robot_ = std::make_unique<TendonRobotModel>(
        config.rod_length,
        config.num_discs,
        config.num_between_nodes,
        config.tendon_input,
        config.K_inv, 
        strain_noise,
        small_wrench_noise_,
        base_pose_mean,
        base_pose_noise);

    get_initial_values();
}


Solution<TendonRobotMarginals> TendonRobotSolver::solve(
    const Vector4Gaussian& tensions,
    const std::optional<Vector6Gaussian>& tip_wrench,
    const std::optional<Vector3Gaussian>& tip_position_meas)
{
    tensions_ = tensions;
    tip_wrench_ = tip_wrench;
    tip_position_meas_ = tip_position_meas;

    Solution<TendonRobotMarginals> solution;
    solution.meta = optimize();
    solution.marginals = extracted_;

    return solution;
}


void TendonRobotSolver::build_graph() {
    // Build base robot graph
    graph_ = robot_->build_graph(tensions_);

    // Constrain all external load wrenches (except base and tip)
    int num_nodes = robot_->get_num_nodes();

    for (int i = 1; i + 1 < num_nodes; ++i) {
        graph_.add(PriorFactor<Vector6>(
            robot_->get_external_wrench_key(i), 
            Vector6::Zero(), 
            small_wrench_noise_));
    }

    // If we have tip force input, use it, otherwise default to tight zero
    Vector6 tip_wrench_mean = Vector6::Zero();
    auto tip_wrench_noise = noiseModel::Gaussian::Covariance(small_wrench_noise_->covariance());
    if (tip_wrench_) {
        tip_wrench_mean = tip_wrench_->mean;
        tip_wrench_noise = noiseModel::Gaussian::Covariance(tip_wrench_->cov);
    }

    graph_.add(PriorFactor<Vector6>(
        robot_->get_external_wrench_key(num_nodes - 1),
        tip_wrench_mean,
        tip_wrench_noise));

    // If we have a tip pose measurement, then use it
    if (tip_position_meas_) {
        graph_.add(PositionPriorFactor(
            robot_->rod_->get_pose_key(-1),
            tip_position_meas_->mean,
            noiseModel::Gaussian::Covariance(tip_position_meas_->cov)));
    }
}


void TendonRobotSolver::extract_solution() {
    extracted_ = robot_->get_marginals(values_, marginals_);
}

void TendonRobotSolver::get_initial_values() {
    values_ = robot_->get_initial_values();
}
