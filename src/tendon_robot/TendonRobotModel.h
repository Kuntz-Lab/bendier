#pragma once

#include <gtsam/base/Vector.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/base/Matrix.h>

#include <optional>

#include "cosserat_rod/CosseratRodModel.h"
#include "utils/Gaussians.h"

// Number of discrete pose nodes for the robot based on model parameters
inline int TendonRobotNumNodes(int num_discs, int num_between_nodes) {
    return num_discs + (num_discs - 1) * num_between_nodes;
}

struct RoutingFunctionParams {
    double angle_offset = 0.0;  // Starting angle (radians)
    double total_angle = 0.0;   // Total angle change across the rod (0 for a straight tendon)
};

struct TendonRoutingInput {
    std::vector<RoutingFunctionParams> params;
    double routing_radius;
};

// TODO I think this might could use restructuring somehow but not sure how yet
struct TendonConfig {
    int num_discs;
    int num_tendons;
    double routing_radius;
    std::vector<int> disc_pose_idx;
    std::vector<int> no_disc_pose_idx;
    std::vector<std::vector<gtsam::Vector3>> hole_locations;  // [disc][tendon]
};

struct TendonRobotMarginals {
    CosseratRodMarginals rod;
    TendonConfig tendon_config;

    std::vector<std::optional<Vector6Gaussian>> external_wrenches;
    VectorXGaussian tensions;
    VectorXGaussian displacements;

    gtsam::Matrix J_pose_tensions;         // 6 x num_tendons
    gtsam::Matrix J_pose_displacements;    // 6 x num_tendons
    gtsam::Matrix J_tension_displacements; // num_tendons x num_tendons
};

class TendonRobotModel {
public:
    using ModelMarginals = TendonRobotMarginals;
    TendonRobotModel(
        double rod_length,
        int num_discs,
        int num_between_nodes,
        TendonRoutingInput tendon_input,
        const gtsam::Matrix6& K_inv,
        gtsam::SharedDiagonal strain_noise,
        gtsam::SharedDiagonal stress_noise,
        gtsam::SharedDiagonal displacement_constraint_noise,
        double tendon_stiffness,
        gtsam::Pose3 base_pose_mean,
        gtsam::SharedDiagonal base_pose_noise,
        const std::vector<int>& external_wrench_node_indices);  // Nodes that have a true external wrench variable

    gtsam::Values get_initial_values() const;

    gtsam::NonlinearFactorGraph build_graph() const;

    std::optional<gtsam::Key> get_external_wrench_key(int node_idx) const;

    gtsam::Key get_tensions_key() const;

    gtsam::Key get_displacements_key() const;

    gtsam::Key get_disc_wrench_key(int disc_idx) const;

    gtsam::Key get_pose_key(int node_idx) const;

    inline int get_num_nodes() const { return num_nodes_; }

    inline int get_num_tendons() const { return num_tendons_; }

    TendonRobotMarginals get_marginals(
        const gtsam::Values& values,
        const gtsam::Marginals& marginals) const;

private:
    void init_tendon_disc_config(TendonRoutingInput tendon_input);

    void compute_reference_lengths();

    void get_J_pose_tensions(const gtsam::Marginals& marginals, TendonRobotMarginals& out) const;

    void get_J_pose_displacements(const gtsam::Marginals& marginals, TendonRobotMarginals& out) const;

    void get_J_tension_displacements(const gtsam::Marginals& marginals, TendonRobotMarginals& out) const;

    std::unique_ptr<CosseratRodModel> rod_;

    const double rod_length_;
    const int num_discs_;
    const int num_nodes_;
    const int num_tendons_;

    gtsam::SharedDiagonal stress_noise_;
    gtsam::SharedDiagonal displacement_constraint_noise_;

    gtsam::Pose3 base_pose_mean_;
    gtsam::SharedDiagonal base_pose_noise_;

    TendonConfig tendon_config_;

    double tendon_stiffness_;
    std::vector<double> reference_lengths_;    // [tendon]

    std::vector<bool> is_external_wrench_node_;
};
