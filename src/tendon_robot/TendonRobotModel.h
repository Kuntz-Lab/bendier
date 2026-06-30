#pragma once

#include <gtsam/base/Vector.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/base/Matrix.h>

#include "cosserat_rod/CosseratRodModel.h"
#include "utils/Gaussians.h"

constexpr int NUM_TENDONS = 4;

enum class RoutingAngleFunction {
    CONSTANT = 0,
    LINEAR = 1
};

struct RoutingFunctionParams {
    double angle_offset = 0.0;  // Starting angle (radians)
    double total_angle = 0.0;   // For LINEAR: total angle change across the rod
};

struct TendonInput {
    std::array<RoutingAngleFunction, NUM_TENDONS> functions;
    std::array<RoutingFunctionParams, NUM_TENDONS> params;
    double routing_radius;
};

struct TendonConfig {
    int num_discs;
    int num_tendons = NUM_TENDONS;
    double routing_radius;
    std::vector<int> disc_pose_idx;
    std::vector<int> no_disc_pose_idx;
    std::vector<std::array<gtsam::Vector3, NUM_TENDONS>> hole_locations;
};

struct TendonRobotMarginals {
    CosseratRodMarginals rod;
    TendonConfig tendon_config;

    std::vector<Vector6Gaussian> external_wrenches;
    Vector4Gaussian tensions;

    Eigen::Matrix<double, 6, NUM_TENDONS> J_pose_tensions;
};

class TendonRobotModel {
public:
    using ModelMarginals = TendonRobotMarginals;
    TendonRobotModel(
        double rod_length,
        int num_discs,
        int num_between_nodes,
        TendonInput tendon_input,
        const gtsam::Matrix6& K_inv,
        gtsam::SharedDiagonal strain_noise,
        gtsam::SharedDiagonal stress_noise,
        gtsam::Pose3 base_pose_mean,
        gtsam::SharedDiagonal base_pose_noise);

    gtsam::Values get_initial_values() const;

    gtsam::NonlinearFactorGraph build_graph() const;

    gtsam::Key get_external_wrench_key(int node_idx) const;

    gtsam::Key get_tensions_key() const;

    gtsam::Key get_disc_wrench_key(int disc_idx) const;

    gtsam::Key get_pose_key(int node_idx) const;

    inline int get_num_nodes() const { return num_nodes_; }

    TendonRobotMarginals get_marginals(
        const gtsam::Values& values,
        const gtsam::Marginals& marginals) const;

private:
    void init_tendon_disc_config(TendonInput tendon_input);

    void get_J_pose_tensions(const gtsam::Marginals& marginals, TendonRobotMarginals& out) const;

    std::unique_ptr<CosseratRodModel> rod_;

    const double rod_length_;
    const int num_discs_;
    const int num_nodes_;

    gtsam::SharedDiagonal stress_noise_;

    gtsam::Pose3 base_pose_mean_;
    gtsam::SharedDiagonal base_pose_noise_;

    TendonConfig tendon_config_;
};
