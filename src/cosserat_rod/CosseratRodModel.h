#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>

#include <numeric>
#include <optional>
#include <vector>

#include "utils/Gaussians.h"

struct CosseratRodState {
    Pose3Gaussian pose;
    Vector6Gaussian stress;  // internal wrench in spatial frame
    std::optional<Vector6Gaussian> wrench;  // external wrench in spatial frame, only present at nodes that were given one
};

struct CosseratRodMarginals {
    std::vector<CosseratRodState> states;
};

inline gtsam::Vector6 StraightRodNominalStrain() {
    gtsam::Vector6 v;
    v << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    return v;
}

inline std::vector<int> AllRodNodes(int num_nodes) {
    // Creates a vector of node indices [0, 1, ..., num_nodes-1]
    std::vector<int> indices(num_nodes);
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
}

struct CosseratRodModelConfig {
    int num_nodes;
    gtsam::Matrix6 K_inv;
    gtsam::SharedDiagonal strain_noise;
    gtsam::SharedDiagonal stress_noise;
    int num_magnus_terms = 4;
    double rod_length = 0.0;
    gtsam::Vector6 nominal_strain = StraightRodNominalStrain();
    std::vector<int> wrench_node_indices = {};
};

class CosseratRodModel {
public:
    using ModelMarginals = CosseratRodMarginals;
    explicit CosseratRodModel(const CosseratRodModelConfig& config);

    gtsam::NonlinearFactorGraph build_graph() const;

    gtsam::Values get_initial_values(
        const gtsam::Pose3& base_pose_init = gtsam::Pose3::Identity()) const;

    CosseratRodMarginals get_marginals(
        const gtsam::Values& values,
        const gtsam::Marginals& marginals) const;

    void set_rod_length(double rod_length);

    void set_nominal_strain(const gtsam::Vector6& nominal_strain);

    inline int get_num_nodes() const { return num_nodes_; }

    gtsam::Key get_pose_key(int node_idx) const;
    gtsam::Key get_stress_key(int node_idx) const;
    gtsam::Key get_wrench_key(int node_idx) const;
    
    bool has_wrench_key(int node_idx) const;

    const std::vector<gtsam::Key>& get_pose_keys() const;

private:
    int clamp_node_idx(int node_idx) const;

    // Unique rod ID for unique GTSAM Symbol keys
    const int id_;
    inline static int next_id_ = 0;

    const int num_nodes_;
    std::vector<gtsam::Matrix6> K_inv_;
    const int num_magnus_terms_;

    gtsam::SharedDiagonal strain_noise_;
    gtsam::SharedDiagonal stress_noise_;

    double rod_length_;
    gtsam::Vector6 nominal_strain_;

    std::vector<gtsam::Key> pose_keys_;
    std::vector<gtsam::Key> stress_keys_;
    std::vector<std::optional<gtsam::Key>> wrench_keys_;
};
