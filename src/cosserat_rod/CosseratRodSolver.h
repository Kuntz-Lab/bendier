#pragma once

#include <gtsam/base/Vector.h>
#include <optional>

#include "cosserat_rod/CosseratRodModel.h"
#include "utils/Gaussians.h"
#include "utils/SolverBase.h"

struct CosseratRodSolverConfig {
    CosseratRodSolverConfig(
        double rod_length,
        int num_nodes,
        const gtsam::Matrix6& K_inv,
        double sigma_constitutive_rot,
        double sigma_constitutive_pos,
        double sigma_equilibrium_force,
        double sigma_equilibrium_moment,
        double sigma_base_pose_pos,
        double sigma_base_pose_rot,
        int num_magnus_terms = 4,
        const SolverBaseConfig& base = {})
    :   base(base),
        rod_length(rod_length),
        num_nodes(num_nodes),
        num_magnus_terms(num_magnus_terms),
        K_inv(K_inv),
        sigma_constitutive_rot(sigma_constitutive_rot),
        sigma_constitutive_pos(sigma_constitutive_pos),
        sigma_equilibrium_force(sigma_equilibrium_force),
        sigma_equilibrium_moment(sigma_equilibrium_moment),
        sigma_base_pose_pos(sigma_base_pose_pos),
        sigma_base_pose_rot(sigma_base_pose_rot)
    {}

    SolverBaseConfig base;

    double rod_length;
    int num_nodes;
    int num_magnus_terms;

    gtsam::Matrix6 K_inv;

    double sigma_constitutive_rot;
    double sigma_constitutive_pos;

    double sigma_equilibrium_force;
    double sigma_equilibrium_moment;

    double sigma_base_pose_pos;
    double sigma_base_pose_rot;
};

class CosseratRodSolver : public SolverBase<CosseratRodModel> {
public:
    CosseratRodSolver(const CosseratRodSolverConfig& config);

    Solution<CosseratRodModel::ModelMarginals> solve(
        const std::optional<Vector6Gaussian>& tip_wrench     = std::nullopt,
        const std::optional<Pose3Gaussian>&   tip_pose       = std::nullopt,
        const std::optional<gtsam::Vector6>&  nominal_strain = std::nullopt);

private:
    gtsam::SharedDiagonal equilibrium_wrench_noise_;
    gtsam::SharedDiagonal base_pose_noise_;
};
