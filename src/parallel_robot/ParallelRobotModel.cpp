#include "ParallelRobotModel.h"

#include <cmath>
#include <gtsam/base/Vector.h>
#include <gtsam/slam/BetweenFactor.h>

#include "PlatformWrenchBalanceFactor.h"
#include "SingleRodBaseFactor.h"
#include "cosserat_rod/CosseratRodModel.h"
#include "utils/Gaussians.h"
#include "utils/MiscInline.h"

using namespace gtsam;


ParallelRobot::ParallelRobot(
    int nodes_per_rod, 
    Matrix6 K_inv,
    SharedDiagonal strain_noise,
    SharedDiagonal stress_noise,
    std::array<Matrix4, NUM_RODS> base_end_poses,
    std::array<Matrix4, NUM_RODS> tip_end_poses,
    double sigma_end_pose_pos,
    double sigma_end_pose_rot)
:
    base_end_poses_(base_end_poses),
    tip_end_poses_(tip_end_poses),
    small_wrench_noise_(stress_noise),
    sigma_end_pose_pos_(sigma_end_pose_pos),
    sigma_end_pose_rot_(sigma_end_pose_rot)
{
    // Make each rod
    for (int i = 0; i < NUM_RODS; i++) {
        rods_[i] = std::make_unique<CosseratRodModel>(
            nodes_per_rod, K_inv, strain_noise, stress_noise);
    }
}


Key platform_pose_key() { return Symbol('P', 424242424242); }


Key platform_wrench_key() { return Symbol('W', 424242424242); }


NonlinearFactorGraph ParallelRobot::build_graph(
    const std::array<double, NUM_RODS>& rod_lengths,
    double sigma_rod_lengths,
    const Vector6Gaussian& wrench_)
{
    NonlinearFactorGraph graph;

    // Tip of the rods relative to platform is relatively certain
    SharedDiagonal tip_pose_noise = get_noise_model_rot_pos(sigma_end_pose_rot_, sigma_end_pose_pos_);

    // Base of rods relative to world is certain, except for z extension, which is uncertain
    gtsam::SharedDiagonal base_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 
        sigma_end_pose_rot_, sigma_end_pose_rot_,
        sigma_end_pose_pos_, sigma_end_pose_pos_, sigma_rod_lengths,
        1.0e-4).finished());

    // Build each rod
    for (int i = 0; i < NUM_RODS; i++) {
        // Build base cosserat rod graph
        graph.add(rods_[i]->build_graph(rod_lengths[i]));

        // Constrain interior wrenches to zero (skip base and tip)
        std::vector<Key> wrench_keys = rods_[i]->get_wrench_keys();
        for (size_t j = 1; j + 1 < wrench_keys.size(); ++j) {
            graph.add(PriorFactor<Vector6>(wrench_keys[j], Vector6::Zero(), small_wrench_noise_));
        }

        // Base pose prior
        graph.add(SingleRodBaseFactor(
            rods_[i]->get_pose_key(0), 
            rods_[i]->get_stress_key(0),
            Pose3(base_end_poses_[i]), 
            base_noise));

        // Tip pose relative to platform
        graph.add(BetweenFactor<Pose3>(
            platform_pose_key(),
            rods_[i]->get_pose_key(-1),
            Pose3(tip_end_poses_[i]),
            tip_pose_noise));
    }
    
    // Put prior on tip wrench based on user input
    graph.add(PriorFactor<Vector6>(
        platform_wrench_key(), 
        wrench_.mean, 
        noiseModel::Gaussian::Covariance(wrench_.cov)));

    // Sum of all transformed tip stresses equals zero (for now)
    graph.add(PlatformWrenchBalanceFactor(
        rods_[0]->get_stress_key(-1),
        rods_[0]->get_pose_key(-1),
        rods_[1]->get_stress_key(-1),
        rods_[1]->get_pose_key(-1),
        rods_[2]->get_stress_key(-1),
        rods_[2]->get_pose_key(-1),
        rods_[3]->get_stress_key(-1),
        rods_[3]->get_pose_key(-1),
        rods_[4]->get_stress_key(-1),
        rods_[4]->get_pose_key(-1),
        rods_[5]->get_stress_key(-1),
        rods_[5]->get_pose_key(-1),
        platform_wrench_key(),
        platform_pose_key(),
        small_wrench_noise_));
    
    return graph;
}


Values ParallelRobot::get_initial_values() const {
    Values values;

    // Values for each rod
    for (int i = 0; i < NUM_RODS; i++) {
        values.insert(rods_[i]->get_initial_values(0.3, Pose3(base_end_poses_[i])));
    }

    // Values for moving platform variables
    values.insert(platform_pose_key(), Pose3(Rot3::Rz(M_PI), Point3(0, 0, 0.6)));
    values.insert(platform_wrench_key(), Vector6(Vector6::Zero()));

    return values;
}


ParallelRobotMarginals ParallelRobot::get_marginals(
    const Values& values, 
    const Marginals& marginals) const
{   
    ParallelRobotMarginals solution;

    for (int i = 0; i < NUM_RODS; i++) {
        solution.rods[i] = rods_[i]->get_marginals(values, marginals);
    }

    solution.platform_pose.mean = values.at<Pose3>(platform_pose_key()).matrix();
    solution.platform_pose.cov = marginals.marginalCovariance(platform_pose_key());

    solution.platform_wrench.mean = values.at<Vector6>(platform_wrench_key()).matrix();
    solution.platform_wrench.cov = marginals.marginalCovariance(platform_wrench_key());
    
    return solution;
}


Matrix6 ParallelRobot::get_rod_lengths_jacobian(const Marginals& marginals) const {
    // Get key vector of all rod base poses
    KeyVector keys;
    for (const auto& rod : rods_) {
        keys.push_back(rod->get_pose_key(0));
    }

    // Put the platform pose (the thing we care about) at the end
    Key T = platform_pose_key();
    keys.push_back(T);
    
    JointMarginal joint = marginals.jointMarginalCovariance(keys);

    // How do the rod lengths vary together?
    Matrix6 sigma_QQ = Matrix6::Zero();  
    for (int i = 0; i < NUM_RODS; ++i) {
        for (int j = 0; j < NUM_RODS; ++j) {
            // Extract z translation component (index 5) from each 6×6 pose block
            sigma_QQ(i, j) = joint(keys[i], keys[j])(5, 5);
        }
    }

    // How is pose correllated with rod lengths?
    Matrix6 sigma_TQ = Matrix6::Zero();
    for (int j = 0; j < NUM_RODS; ++j) {
        // 6×6 covariance between platform pose and rod base pose
        Matrix6 block = joint(T, keys[j]);

        // Extract column corresponding to rod length (z component)
        sigma_TQ.col(j) = block.col(5);
    }

    // Solve, see paper for why this works.
    Eigen::LDLT<Matrix6> ldlt(sigma_QQ);
    return sigma_TQ * ldlt.solve(Matrix6::Identity());
}


Matrix6 ParallelRobot::get_tip_wrench_jacobian(const Marginals& marginals) const {
    // Get joint marginal between tip wrench and tip pose
    Key W = platform_wrench_key();
    Key T = platform_pose_key();

    KeyVector keys;
    keys.push_back(W);
    keys.push_back(T);
    JointMarginal joint = marginals.jointMarginalCovariance(keys);

    // Get individual blocks
    Matrix6 sigma_WW = joint(W, W);
    Matrix6 sigma_TW = joint(T, W);

    // Compute J_pose_tensions = sigma_TQ * inv(sigma_QQ)
    Eigen::LDLT<Eigen::MatrixXd> ldlt(sigma_WW);
    return sigma_TW * ldlt.solve(Matrix6::Identity());
}