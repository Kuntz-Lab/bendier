#pragma once

#include <gtsam/base/Vector.h>

#include "cosserat_rod/CosseratRodModel.h"
#include "utils/Gaussians.h"
#include "utils/SolverBase.h"


struct CosseratRodSolverConfig {
    SolverBaseConfig base;
    
    double rod_length;
    int num_nodes;
    int num_magnus_terms;
    
    gtsam::Matrix6 K_inv;

    double sigma_strain_pos;
    double sigma_strain_rot;

    double sigma_small_force;
    double sigma_small_moment;

    double sigma_base_pose_pos;
    double sigma_base_pose_rot;
};


class CosseratRodSolver : public SolverBase {
public:
    CosseratRodSolver(const CosseratRodSolverConfig& config);

    Solution<CosseratRodMarginals> solve(
        const std::optional<Vector6Gaussian>& tip_wrench,
        const std::optional<Pose3Gaussian>& tip_pose,
        const std::optional<gtsam::Vector6>& nominal_strain);

private:
    void build_graph() override;

    void extract_solution() override;

    void get_initial_values() override;
    
    std::optional<Vector6Gaussian> tip_wrench_;
    std::optional<Pose3Gaussian> tip_pose_;
    std::optional<gtsam::Vector6> nominal_strain_;

    double rod_length_;
    
    gtsam::SharedDiagonal small_wrench_noise_;
    gtsam::SharedDiagonal base_pose_noise_;

    std::unique_ptr<CosseratRodModel> rod_;
    CosseratRodMarginals extracted_;
};