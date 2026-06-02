#include "CosseratRodSolver.h"

#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>

#include "cosserat_rod/CosseratRodModel.h"
#include "utils/Gaussians.h"
#include "utils/MiscInline.h"
#include "utils/SolverBase.h"

using namespace gtsam;


CosseratRodSolver::CosseratRodSolver(const CosseratRodSolverConfig& config)
:   
    SolverBase(config.base),
    rod_length_(config.rod_length)
{
    SharedDiagonal strain_noise = get_noise_model_rot_pos(
        config.sigma_strain_rot, config.sigma_strain_pos); 
    
    small_wrench_noise_ = get_noise_model_rot_pos(
        config.sigma_small_moment, config.sigma_small_force); 
    
    base_pose_noise_ = get_noise_model_rot_pos(
        config.sigma_base_pose_rot, config.sigma_base_pose_pos);

    rod_= std::make_unique<CosseratRodModel>(
        config.num_nodes, 
        config.K_inv, 
        strain_noise, 
        small_wrench_noise_,
        config.num_magnus_terms);

    get_initial_values();
}


void CosseratRodSolver::extract_solution() {
    extracted_ = rod_->get_marginals(values_, marginals_);
}



void CosseratRodSolver::get_initial_values() {
    values_ = rod_->get_initial_values(rod_length_);
}


Solution<CosseratRodMarginals> CosseratRodSolver::solve(
    const std::optional<Vector6Gaussian>& tip_wrench, 
    const std::optional<Pose3Gaussian>& tip_pose,
    const std::optional<Vector6>& nominal_strain) 
{
    tip_wrench_ = tip_wrench; 
    tip_pose_ = tip_pose;
    nominal_strain_ = nominal_strain;

    Solution<CosseratRodMarginals> solution;
    solution.meta = optimize();
    solution.marginals = extracted_;

    return solution;
}


void CosseratRodSolver::build_graph() {
    // Build base rod graph
    graph_ = rod_->build_graph(rod_length_, nominal_strain_);

    // Constrain base pose to identity 
    graph_.add(PriorFactor<Pose3>(
        rod_->get_pose_key(0), 
        Pose3::Identity(), 
        base_pose_noise_));

    // Constrain all wrenches on the interior of the rod to be zero
    std::vector<Key> wrench_keys = rod_->get_wrench_keys();

    // Skip base and tip wrenches
    for (size_t i = 1; i + 1 < wrench_keys.size(); ++i) {
        graph_.add(PriorFactor<Vector6>(
            wrench_keys[i],
            Vector6::Zero(),
            small_wrench_noise_));
    }

    // Set prior on tip wrench/pose based on user input
    if (tip_wrench_) {
        graph_.add(PriorFactor<Vector6>(
            wrench_keys.back(),
            (*tip_wrench_).mean,
            noiseModel::Gaussian::Covariance((*tip_wrench_).cov)));
    }

    if (tip_pose_) {
        graph_.add(PriorFactor<Pose3>(
            rod_->get_pose_key(-1),
            Pose3((*tip_pose_).mean),
            noiseModel::Gaussian::Covariance((*tip_pose_).cov)));
    }
}
