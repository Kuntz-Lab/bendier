#pragma once

#include <optional>

#include "utils/Gaussians.h"
#include "utils/SolverBase.h"
#include "TendonRobotModel.h"
#include <gtsam/linear/NoiseModel.h>

struct TendonRobotSolverConfig {
    TendonRobotSolverConfig(
        double rod_length,
        int num_discs,
        int num_between_nodes,
        const gtsam::Matrix6& K_inv,
        double sigma_strain_rot,
        double sigma_strain_pos,
        double sigma_small_force,
        double sigma_small_moment,
        double sigma_base_pose_pos,
        double sigma_base_pose_rot,
        const TendonInput& tendon_input,
        double sigma_displacement_constraint = 1e-6,
        const std::vector<double>& axial_stiffness = {},
        double sigma_tension_nonneg = 0.1,
        const SolverBaseConfig& base = {})
    :   base(base),
        rod_length(rod_length),
        num_discs(num_discs),
        num_between_nodes(num_between_nodes),
        K_inv(K_inv),
        sigma_strain_rot(sigma_strain_rot),
        sigma_strain_pos(sigma_strain_pos),
        sigma_small_force(sigma_small_force),
        sigma_small_moment(sigma_small_moment),
        sigma_base_pose_pos(sigma_base_pose_pos),
        sigma_base_pose_rot(sigma_base_pose_rot),
        sigma_displacement_constraint(sigma_displacement_constraint),
        axial_stiffness(axial_stiffness),
        tendon_input(tendon_input)
    {}

    SolverBaseConfig base;

    double rod_length;
    int num_discs;
    int num_between_nodes;
    gtsam::Matrix6 K_inv;

    double sigma_strain_rot;
    double sigma_strain_pos;
    double sigma_small_force;
    double sigma_small_moment;
    double sigma_base_pose_pos;
    double sigma_base_pose_rot;

    // Tightness of the displacement-constraint factor 
    // Validity of:  displacement == predicted(poses, tensions)
    double sigma_displacement_constraint;

    // Axial stiffness (EA) per tendon; empty defaults to a rigid tendon.
    std::vector<double> axial_stiffness; // TODO rename

    TendonInput tendon_input;
};

class TendonRobotSolver : public SolverBase<TendonRobotModel> {
public:
    TendonRobotSolver(const TendonRobotSolverConfig& config);

    Solution<TendonRobotModel::ModelMarginals> solve(
        const VectorXGaussian&                tensions,
        const std::optional<Vector6Gaussian>& tip_wrench         = std::nullopt,
        const std::optional<Vector3Gaussian>& tip_position_meas  = std::nullopt,
        const std::optional<VectorXGaussian>& displacement_meas  = std::nullopt);

private:
    gtsam::SharedDiagonal small_wrench_noise_;
};
