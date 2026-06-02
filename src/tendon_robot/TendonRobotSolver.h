#pragma once

#include "utils/Gaussians.h"
#include "utils/SolverBase.h"
#include "TendonRobotModel.h"
#include <gtsam/linear/NoiseModel.h>


struct TendonRobotSolverConfig{
    SolverBaseConfig base;
    
    double rod_length;
    int num_discs;
    int num_between_nodes;
    gtsam::Matrix6 K_inv;

    double sigma_strain_rot;
    double sigma_strain_pos;
    double sigma_stress_force;
    double sigma_stress_moment;
    double sigma_base_pos;
    double sigma_base_rot;
    
    TendonInput tendon_input;
};


class TendonRobotSolver : SolverBase {
public:
    TendonRobotSolver(const TendonRobotSolverConfig& config);

    Solution<TendonRobotMarginals> solve(
        const Vector4Gaussian& tensions,
        const std::optional<Vector6Gaussian>& tip_wrench,
        const std::optional<Vector3Gaussian>& tip_position_meas);

private:
    void build_graph() override;

    void extract_solution() override;

    void get_initial_values() override;

    gtsam::SharedDiagonal small_wrench_noise_;

    std::unique_ptr<TendonRobotModel> robot_;
    
    Vector4Gaussian tensions_;
    std::optional<Vector6Gaussian> tip_wrench_;
    std::optional<Vector3Gaussian> tip_position_meas_;

    TendonRobotMarginals extracted_;
};


    // std::vector<Vector> sample_cov(const Matrix& cov, int num_samples) {
    //     const int dim = cov.rows();
    //     Eigen::LLT<Matrix> llt(cov);
    //     Matrix L = llt.matrixL();

    //     static std::random_device rd;
    //     static std::mt19937 gen(rd());
    //     static std::normal_distribution<> normal(0.0, 1.0);  // N(0,1)

    //     std::vector<Vector> samples;
    //     samples.reserve(num_samples);

    //     for (int n = 0; n < num_samples; ++n) {
    //         Vector z(dim);
    //         for (int i = 0; i < dim; ++i)
    //             z(i) = normal(gen);

    //         Vector delta = L * z;
    //         samples.push_back(delta);
    //     }

    //     return samples;
    // }

    // void sample_tip_pose(TendonRobotSolution& solution, int num_samples) {
    //     Pose3 tip_pose_mean = Pose3(solution.backbone_pose_mean.back());
    //     Matrix6 tip_pose_cov = solution.backbone_pose_cov.back();

    //     std::vector<Vector> d_tip_pose = sample_cov(tip_pose_cov, num_samples);
    //     d_tip_pose.reserve(num_samples);

    //     for (int i = 0; i < num_samples; i++) {
    //         solution.tip_pose_samples[i] = tip_pose_mean.retract(d_tip_pose[i]).matrix();
    //     }
    // }

    // void sample_fbg_array(TendonRobotSolution& solution, int num_samples) {   
    //     KeyVector stress_keys;
    //     stress_keys.reserve(num_backbone_poses_);

    //     for (int i = 0; i < num_backbone_poses_; ++i) {
    //         stress_keys.push_back(S(i));
    //     }

    //     Matrix joint_stress_cov = marginals_.jointMarginalCovariance(stress_keys).fullMatrix();
    //     std::vector<Vector> joint_d_stresses = sample_cov(joint_stress_cov, num_samples);
        
    //     for (int i = 0; i < num_samples; ++i) {
    //         std::vector<Vector3> fbg_array_sample;
    //         fbg_array_sample.reserve(num_backbone_poses_);

    //         for (int j = 0; j < num_backbone_poses_; ++j) {
    //             Vector6 stress_mean = values_.at<Vector6>(S(j));

    //             Vector6 d_stress = joint_d_stresses[i].segment<6>(6 * j);
    //             Vector6 stress = stress_mean + d_stress;

    //             fbg_array_sample.push_back(stress_to_fbg_signal(stress, K_inv_, rod_diameter_));
    //         }
    //         solution.fbg_array_samples[i] = fbg_array_sample;
    //     }
    // }

    // void sample_solution(TendonRobotSolution& solution, int num_samples) 
    // {
    //     sample_tip_pose(solution, num_samples);
    //     sample_fbg_array(solution, num_samples);
    // }
        

// class TipForceSolver : public TendonRobotGtsam {
// public:
//     TipForceSolver(const TendonRobotConfig& config) 
//         : TendonRobotGtsam(config)
//     {
//         tip_wrench_cov_ = noiseModel::Diagonal::Sigmas((Vector(6) << 
//             config.small_moment_std, config.small_moment_std, config.small_moment_std, 
//             config.tip_force_prior_std, config.tip_force_prior_std, config.tip_force_prior_std).finished());
            
//         tip_position_meas_cov_ = noiseModel::Isotropic::Sigma(3, config.tip_position_meas_std);

//         last_tip_wrench_mean_ = Vector6::Zero();
//         last_tip_wrench_cov_ = 10 * Matrix6::Identity();
//         tip_wrench_drift_cov_ = config.tip_force_drift_std * config.tip_force_drift_std * Matrix6::Identity();
//     }

// private:
//     noiseModel::Diagonal::shared_ptr tip_position_meas_cov_;
//     noiseModel::Diagonal::shared_ptr tip_wrench_cov_;

//     Vector6 last_tip_wrench_mean_;
//     Matrix6 last_tip_wrench_cov_;
//     Matrix6 tip_wrench_drift_cov_;

//     void add_common_factors() {
//         // Applied wrenches are all zero, exect at the tip
//         for (int i = 1; i + 1 < num_backbone_poses_; i++) {
//             graph_.add(PriorFactor<Vector6>(F(i), Vector6::Zero(), small_wrench_cov_));
//         }
//     }

// public:
//     TendonRobotSolution step(const Vector4& tensions_meas, const Vector3& tip_position_meas, int num_samples) {
//         build_graph_base(tensions_meas);
//         add_common_factors();

//         // Tip force drift prior
//         graph_.add(PriorFactor<Vector6>(F(num_backbone_poses_ - 1), last_tip_wrench_mean_, 
//             noiseModel::Gaussian::Covariance(last_tip_wrench_cov_ + tip_wrench_drift_cov_)));

//         // Tip force prior is zero with big uncertainty
//         graph_.add(PriorFactor<Vector6>(F(num_backbone_poses_ - 1), Vector6::Zero(), tip_wrench_cov_));

//         // Tip pose measurement prior
//         graph_.add(PositionMeasurementFactor(T(num_backbone_poses_ - 1), tip_position_meas, tip_position_meas_cov_));

//         // Run the optimizer, etc
//         TendonRobotSolution solution = update(num_samples);

//         last_tip_wrench_mean_ = solution.applied_wrench_mean.back();
//         last_tip_wrench_cov_ = solution.applied_wrench_cov.back();

//         return solution;
//     }

//     TendonRobotSolution simulation_step(const Vector4& tensions, const Vector3& tip_force) {
//         build_graph_base(tensions);
//         add_common_factors();
        
//         // Known tip force factor
//         Vector6 tip_wrench_mean;
//         tip_wrench_mean.head<3>() = Vector3::Zero();
//         tip_wrench_mean.tail<3>() = tip_force;
//         graph_.add(PriorFactor<Vector6>(F(num_backbone_poses_ - 1), tip_wrench_mean, small_wrench_cov_));
        
//         TendonRobotSolution solution = update(1);

//         return solution;
//     }
// };


// class DistLoadSolver : public TendonRobotGtsam {
// public:
//     DistLoadSolver(const TendonRobotConfig& config) 
//         : TendonRobotGtsam(config)
//     {
//         dist_load_prior_cov_ = noiseModel::Diagonal::Sigmas((Vector(6) << 
//             config.small_moment_std, config.small_moment_std, config.small_moment_std, 
//             config.dist_load_prior_std, config.dist_load_prior_std, config.dist_load_prior_std).finished());
        
//         dist_load_smoothing_cov_ = noiseModel::Isotropic::Sigma(3, config.dist_load_smoothness_std);

//         fbg_strain_meas_cov_ = noiseModel::Isotropic::Sigma(3, config.fbg_strain_meas_std);

//         for (int i = 1; i < num_backbone_poses_; ++i) {
//             last_wrenches_mean_.push_back(Vector6::Zero());
//             last_wrenches_cov_.push_back(10 * Matrix6::Identity());
//         }
        
//         wrenches_drift_cov_ = config.dist_load_drift_std * config.dist_load_drift_std * Matrix6::Identity();
//     }

// private:
//     noiseModel::Diagonal::shared_ptr dist_load_prior_cov_;
//     noiseModel::Isotropic::shared_ptr dist_load_smoothing_cov_;
//     noiseModel::Isotropic::shared_ptr fbg_strain_meas_cov_;

//     std::vector<Vector6> last_wrenches_mean_;
//     std::vector<Matrix6> last_wrenches_cov_;
//     Matrix6 wrenches_drift_cov_;

// public:
//     TendonRobotSolution step(const Vector4& tensions_meas, const std::vector<Vector3>& fbg_signals_meas, int num_samples) {
//         build_graph_base(tensions_meas);

//         // Priors for wrenches
//         for (int i = 1; i < num_backbone_poses_; i++) {
//             // Magnitude prior factors for distributed load
//             graph_.add(PriorFactor<Vector6>(F(i), Vector6::Zero(), dist_load_prior_cov_));

//             // Drift prior factors
//             graph_.add(PriorFactor<Vector6>(F(i), last_wrenches_mean_[i - 1], 
//                 noiseModel::Gaussian::Covariance(last_wrenches_cov_[i - 1] + wrenches_drift_cov_)));
//         }

//         // Smoothing prior factors for distributed load
//         for (int i = 1; i + 3 < num_backbone_poses_; ++i) {
//             graph_.add(DistLoadSmoothingFactor(F(i), F(i + 1), F(i + 2), F(i + 3), dist_load_smoothing_cov_));
//         }

//         // FBG strain measurement factors
//         for (int i = 0; i < num_backbone_poses_; ++i) {
//             graph_.add(FbgMeasurementFactor(S(i), fbg_signals_meas[i], K_inv_, rod_diameter_, fbg_strain_meas_cov_));
//         }

//         TendonRobotSolution solution = update(num_samples);

//         last_wrenches_mean_ = solution.applied_wrench_mean;
//         last_wrenches_cov_ = solution.applied_wrench_cov;

//         return solution;
//     }

//     TendonRobotSolution step_simulation(const Vector4& tensions, const std::vector<Vector3>& forces) {
//         build_graph_base(tensions);

//         // Add applied loads with small uncertainty
//         for (int i = 1; i < num_backbone_poses_; i++) {
//             Vector6 applied_wrench;
//             applied_wrench.head<3>() = Vector3::Zero();
//             applied_wrench.tail<3>() = forces[i - 1];
//             graph_.add(PriorFactor<Vector6>(F(i), applied_wrench, small_wrench_cov_));
//         }

//         TendonRobotSolution solution = update(1);

//         return solution;
//     }
// };
// }


