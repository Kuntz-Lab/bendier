#include "ParallelRobotModel.h"

#include <cmath>
#include <stdexcept>
#include <gtsam/base/Vector.h>
#include <gtsam/slam/BetweenFactor.h>

#include "PlatformWrenchBalanceFactor.h"
#include "SingleRodBaseFactor.h"
#include "cosserat_rod/CosseratRodModel.h"
#include "utils/Gaussians.h"
#include "utils/MiscInline.h"
#include "utils/ModelConcept.h"

using namespace gtsam;

static_assert(BendierModel<ParallelRobotModel>);

ParallelRobotModel::ParallelRobotModel(
    int nodes_per_rod,
    Matrix6 K_inv,
    SharedDiagonal strain_noise,
    SharedDiagonal stress_noise,
    std::vector<Matrix4> base_end_poses,
    std::vector<Matrix4> tip_end_poses,
    double sigma_end_pose_pos,
    double sigma_end_pose_rot,
    double sigma_rod_lengths)
:
    base_end_poses_(std::move(base_end_poses)),
    tip_end_poses_(std::move(tip_end_poses)),
    small_wrench_noise_(stress_noise),
    id_(next_id_++),
    sigma_end_pose_pos_(sigma_end_pose_pos),
    sigma_end_pose_rot_(sigma_end_pose_rot),
    sigma_rod_lengths_(sigma_rod_lengths)
{
    if (base_end_poses_.size() != tip_end_poses_.size())
        throw std::invalid_argument(
            "ParallelRobotModel: base_end_poses and tip_end_poses must be the same size");

    const int num_rods = static_cast<int>(base_end_poses_.size());
    rods_.reserve(num_rods);
    for (int i = 0; i < num_rods; i++) {
        // Only the base and tip nodes of each rod get a wrench variable.
        // Assumes no forces along interior of rods, which could be added later if needed.
        rods_.push_back(std::make_unique<CosseratRodModel>(CosseratRodModelConfig{
            .num_nodes = nodes_per_rod,
            .K_inv = K_inv,
            .strain_noise = strain_noise,
            .stress_noise = stress_noise,
            .wrench_node_indices = {0, nodes_per_rod - 1},
        }));
    }
}

void ParallelRobotModel::set_rod_lengths(const Vector& rod_lengths) {
    rod_lengths_ = rod_lengths;
    for (size_t i = 0; i < rods_.size(); i++)
        rods_[i]->set_rod_length(rod_lengths[i]);
}

void ParallelRobotModel::set_sigma_rod_lengths(double sigma_rod_lengths) {
    sigma_rod_lengths_ = sigma_rod_lengths;
}

Key ParallelRobotModel::get_rod_wrench_key(int rod_idx, int node_idx) const {
    return rods_[rod_idx]->get_wrench_key(node_idx);
}

Key ParallelRobotModel::platform_pose_key() const { return Symbol('P', id_); }

Key ParallelRobotModel::platform_wrench_key() const { return Symbol('W', id_); }

NonlinearFactorGraph ParallelRobotModel::build_graph() const
{
    NonlinearFactorGraph graph;

    SharedDiagonal tip_pose_noise = get_noise_model_rot_pos(sigma_end_pose_rot_, sigma_end_pose_pos_);

    SharedDiagonal base_noise = noiseModel::Diagonal::Sigmas((Vector(6) <<
        sigma_end_pose_rot_, sigma_end_pose_rot_,
        sigma_end_pose_pos_, sigma_end_pose_pos_, sigma_rod_lengths_,
        1.0e-4).finished());

    KeyVector tip_stress_keys;
    KeyVector tip_pose_keys;
    tip_stress_keys.reserve(rods_.size());
    tip_pose_keys.reserve(rods_.size());

    for (size_t i = 0; i < rods_.size(); i++) {
        graph.add(rods_[i]->build_graph());

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

        tip_stress_keys.push_back(rods_[i]->get_stress_key(-1));
        tip_pose_keys.push_back(rods_[i]->get_pose_key(-1));
    }

    // Sum of all transformed tip stresses equals the platform wrench
    graph.add(PlatformWrenchBalanceFactor(
        tip_stress_keys,
        tip_pose_keys,
        platform_wrench_key(),
        platform_pose_key(),
        small_wrench_noise_));

    return graph;
}

Values ParallelRobotModel::get_initial_values() const {
    Values values;

    for (size_t i = 0; i < rods_.size(); i++) {
        values.insert(rods_[i]->get_initial_values(Pose3(base_end_poses_[i])));
    }

    // Kind of sloppy here but it works
    double mean_rod_length = 0;
    for (double l : rod_lengths_) mean_rod_length += l / static_cast<double>(rods_.size());
    values.insert(platform_pose_key(), Pose3(Rot3::Rz(M_PI), Point3(0, 0, mean_rod_length)));
    values.insert(platform_wrench_key(), Vector6(Vector6::Zero()));

    return values;
}

ParallelRobotMarginals ParallelRobotModel::get_marginals(
    const Values& values,
    const Marginals& marginals) const
{
    ParallelRobotMarginals solution;

    solution.rods.resize(rods_.size());
    for (size_t i = 0; i < rods_.size(); i++) {
        solution.rods[i] = rods_[i]->get_marginals(values, marginals);
    }

    solution.platform_pose.mean = values.at<Pose3>(platform_pose_key()).matrix();
    solution.platform_pose.cov = marginals.marginalCovariance(platform_pose_key());

    solution.platform_wrench.mean = values.at<Vector6>(platform_wrench_key()).matrix();
    solution.platform_wrench.cov = marginals.marginalCovariance(platform_wrench_key());

    solution.rod_lengths_jacobian = get_rod_lengths_jacobian(marginals);
    solution.tip_wrench_jacobian  = get_tip_wrench_jacobian(marginals);

    return solution;
}

// NOTE: this proxies "rod length" with the base pose's own z-translation
// (see sigma_QQ/sigma_TQ below), which models a longer commanded rod
// length as the base sliding further away along its own axis rather than
// the rod itself becoming physically longer. A physically longer rod is
// more compliant (it bends more under the same load/stiffness), which
// this proxy does not capture, so this Jacobian is measurably (~1%, more
// under load) off from the true finite-difference sensitivity of platform
// pose to commanded rod length. See RobotJacobianTests.cpp.
Matrix ParallelRobotModel::get_rod_lengths_jacobian(const Marginals& marginals) const {
    const int num_rods = static_cast<int>(rods_.size());

    KeyVector keys;
    for (const auto& rod : rods_) {
        keys.push_back(rod->get_pose_key(0));
    }

    Key T = platform_pose_key();
    keys.push_back(T);

    JointMarginal joint = marginals.jointMarginalCovariance(keys);

    // How do the rod lengths vary together?
    Matrix sigma_QQ = Matrix::Zero(num_rods, num_rods);
    for (int i = 0; i < num_rods; ++i) {
        for (int j = 0; j < num_rods; ++j) {
            sigma_QQ(i, j) = joint(keys[i], keys[j])(5, 5);
        }
    }

    // How is platform pose correlated with rod lengths?
    Matrix sigma_TQ = Matrix::Zero(6, num_rods);
    for (int j = 0; j < num_rods; ++j) {
        Matrix6 block = joint(T, keys[j]);
        sigma_TQ.col(j) = block.col(5);
    }

    // Solve, see paper for why this works.
    Eigen::LDLT<Matrix> ldlt(sigma_QQ);
    return sigma_TQ * ldlt.solve(Matrix::Identity(num_rods, num_rods));
}

Matrix6 ParallelRobotModel::get_tip_wrench_jacobian(const Marginals& marginals) const {
    Key W = platform_wrench_key();
    Key T = platform_pose_key();

    KeyVector keys;
    keys.push_back(W);
    keys.push_back(T);
    JointMarginal joint = marginals.jointMarginalCovariance(keys);

    Matrix6 sigma_WW = joint(W, W);
    Matrix6 sigma_TW = joint(T, W);

    Eigen::LDLT<Eigen::MatrixXd> ldlt(sigma_WW);
    return sigma_TW * ldlt.solve(Matrix6::Identity());
}
