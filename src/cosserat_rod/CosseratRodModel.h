#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>

#include "utils/Gaussians.h"

struct CosseratRodState {
    Pose3Gaussian pose;
    Vector6Gaussian stress;
    Vector6Gaussian wrench;
};

struct CosseratRodMarginals {
    std::vector<CosseratRodState> states;
};

inline gtsam::Vector6 DefaultNominalStrain() {
    gtsam::Vector6 v;
    v << 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    return v;
}

class CosseratRodModel {
public:
    using ModelMarginals = CosseratRodMarginals;
    CosseratRodModel(
        int num_nodes,
        const gtsam::Matrix6& K_inv,
        gtsam::SharedDiagonal strain_noise,
        gtsam::SharedDiagonal stress_noise,
        int num_magnus_terms = 4,
        double rod_length = 0.0,
        const gtsam::Vector6& nominal_strain = DefaultNominalStrain());

    gtsam::NonlinearFactorGraph build_graph() const;

    gtsam::Values get_initial_values(
        const gtsam::Pose3& base_pose_init = gtsam::Pose3::Identity()) const;

    CosseratRodMarginals get_marginals(
        const gtsam::Values& values,
        const gtsam::Marginals& marginals) const;

    void set_rod_length(double rod_length);
    
    void set_nominal_strain(const gtsam::Vector6& nominal_strain);

    gtsam::Key get_pose_key(int node_idx) const;
    gtsam::Key get_stress_key(int node_idx) const;
    gtsam::Key get_wrench_key(int node_idx) const;
    const std::vector<gtsam::Key>& get_wrench_keys() const;
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
    std::vector<gtsam::Key> wrench_keys_;
};
