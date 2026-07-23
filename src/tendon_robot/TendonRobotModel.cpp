#include "TendonRobotModel.h"
#include "cosserat_rod/CosseratRodModel.h"
#include "utils/ModelConcept.h"

#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <memory>

#include "TendonDiscWrenchFactor.h"
#include "TendonDisplacementFactor.h"
#include "utils/Gaussians.h"

using namespace gtsam;

static_assert(BendierModel<TendonRobotModel>);

TendonRobotModel::TendonRobotModel(
    double rod_length,
    int num_discs,
    int num_between_nodes,
    TendonRoutingInput tendon_input,
    const Matrix6& K_inv,
    SharedDiagonal strain_noise,
    SharedDiagonal stress_noise,
    SharedDiagonal displacement_constraint_noise,
    double tendon_stiffness,
    Pose3 base_pose_mean,
    SharedDiagonal base_pose_noise,
    const std::vector<int>& external_wrench_node_indices)
:
    rod_length_(rod_length),
    num_discs_(num_discs),
    num_nodes_(TendonRobotNumNodes(num_discs, num_between_nodes)),
    num_tendons_(static_cast<int>(tendon_input.params.size())),
    stress_noise_(stress_noise),
    displacement_constraint_noise_(displacement_constraint_noise),
    base_pose_mean_(base_pose_mean),
    base_pose_noise_(base_pose_noise),
    tendon_stiffness_(tendon_stiffness)
{
    init_tendon_disc_config(tendon_input);
    compute_reference_lengths();

    // Determine which nodes have a true external wrench variable (other than disc wrenches)
    is_external_wrench_node_.assign(num_nodes_, false);
    for (int idx : external_wrench_node_indices)
        is_external_wrench_node_[idx] = true;
    
    // The underlying cosserat rod will have a wrench at nodes that either have a disc or are a true external wrench node
    std::vector<bool> rod_wants_wrench = is_external_wrench_node_;
    for (size_t disc_idx = 1; disc_idx < tendon_config_.disc_pose_idx.size(); ++disc_idx)
        rod_wants_wrench[tendon_config_.disc_pose_idx[disc_idx]] = true;
    
    // Extract the indices of the rod nodes that will have a wrench variable
    std::vector<int> rod_wrench_node_indices;
    for (int i = 0; i < num_nodes_; ++i)
        if (rod_wants_wrench[i])
            rod_wrench_node_indices.push_back(i);

    rod_ = std::make_unique<CosseratRodModel>(CosseratRodModelConfig{
        .num_nodes = num_nodes_,
        .K_inv = K_inv,
        .strain_noise = strain_noise,
        .stress_noise = stress_noise,
        .rod_length = rod_length_,
        .wrench_node_indices = rod_wrench_node_indices,
    });
}

void TendonRobotModel::init_tendon_disc_config(TendonRoutingInput routing) {
    tendon_config_.num_discs = num_discs_;
    tendon_config_.num_tendons = num_tendons_;
    tendon_config_.disc_pose_idx.reserve(num_discs_);
    tendon_config_.routing_radius = routing.routing_radius;
    tendon_config_.hole_locations.reserve(num_discs_);

    // Compute normalized arc-length positions for poses and discs
    std::vector<double> pose_s(num_nodes_);
    std::vector<double> disc_s(num_discs_);

    for (int i = 0; i < num_nodes_; ++i)
        pose_s[i] = static_cast<double>(i) / (num_nodes_ - 1);

    for (int i = 0; i < num_discs_; ++i)
        disc_s[i] = static_cast<double>(i) / (num_discs_ - 1);

    // For each disc, find the closest pose index
    for (int disc_idx = 0; disc_idx < num_discs_; ++disc_idx) {
        double s = disc_s[disc_idx];

        // Find closest pose index to this disc
        int closest_pose_idx = static_cast<int>(std::round(s * (num_nodes_ - 1)));

        tendon_config_.disc_pose_idx.push_back(closest_pose_idx);
        std::vector<Vector3> holes(num_tendons_);

        for (int tendon_idx = 0; tendon_idx < num_tendons_; ++tendon_idx) {
            double theta = routing.params[tendon_idx].angle_offset
                + s * routing.params[tendon_idx].total_angle;

            double x = routing.routing_radius * std::cos(theta);
            double y = routing.routing_radius * std::sin(theta);
            double z = 0.0;

            holes[tendon_idx] = Vector3(x, y, z);
        }

        tendon_config_.hole_locations.push_back(holes);
    }

    // Find all nodes that aren't associated with discs, useful later
    std::vector<bool> is_disc(num_nodes_, false);

    for (int idx : tendon_config_.disc_pose_idx)
        is_disc[idx] = true;

    for (int i = 0; i < num_nodes_; ++i)
        if (!is_disc[i])
            tendon_config_.no_disc_pose_idx.push_back(i);
}

void TendonRobotModel::compute_reference_lengths() {
    // Geometric tendon length in the straight, untwisted reference
    // Note this will NOT work for precurved backbones or otherwise.
    reference_lengths_.assign(num_tendons_, 0.0);

    for (int disc_idx = 0; disc_idx + 1 < num_discs_; ++disc_idx) {
        double s0 = static_cast<double>(disc_idx) / (num_discs_ - 1);
        double s1 = static_cast<double>(disc_idx + 1) / (num_discs_ - 1);
        Vector3 dz(0.0, 0.0, rod_length_ * (s1 - s0));

        const std::vector<Vector3>& holes0 = tendon_config_.hole_locations[disc_idx];
        const std::vector<Vector3>& holes1 = tendon_config_.hole_locations[disc_idx + 1];

        for (int i = 0; i < num_tendons_; ++i)
            reference_lengths_[i] += ((holes1[i] - holes0[i]) + dz).norm();
    }
}

Key TendonRobotModel::get_pose_key(int node_idx) const {
    return rod_->get_pose_key(node_idx);
}

Key TendonRobotModel::get_tensions_key() const {
    return Symbol('Q', 424242);
}

Key TendonRobotModel::get_displacements_key() const {
    return Symbol('X', 424242);
}

Key TendonRobotModel::get_disc_wrench_key(int disc_idx) const {
    // We dont ever want to include disc wrenches for base disc
    if (disc_idx < 1)
        throw std::out_of_range("TendonRobot: invalid disc wrench index");

    return Symbol('D', disc_idx);
}

std::optional<Key> TendonRobotModel::get_external_wrench_key(int node_idx) const {
    if (!is_external_wrench_node_[node_idx])
        return std::nullopt;

    // A non-base disc has its own dedicated external-wrench variable,
    // separate from the rod's own wrench key at that node (which is fully
    // determined by TendonDiscWrenchFactor's tendon-force equation).
    for (size_t disc_idx = 1; disc_idx < tendon_config_.disc_pose_idx.size(); ++disc_idx) {
        if (tendon_config_.disc_pose_idx[disc_idx] == node_idx) {
            return get_disc_wrench_key(disc_idx);
        }
    }

    // Otherwise (base disc, or a genuinely non-disc node), the rod's own
    // wrench key directly represents the external wrench.
    return rod_->get_wrench_key(node_idx);
}

Values TendonRobotModel::get_initial_values() const {
    Values values;

    values.insert(rod_->get_initial_values());
    values.insert(get_tensions_key(), Vector(Vector::Zero(num_tendons_)));
    values.insert(get_displacements_key(), Vector(Vector::Zero(num_tendons_)));

    for (size_t disc_idx = 1; disc_idx < tendon_config_.disc_pose_idx.size(); ++disc_idx) {
        if (is_external_wrench_node_[tendon_config_.disc_pose_idx[disc_idx]])
            values.insert(get_disc_wrench_key(disc_idx), Vector6(Vector6::Zero()));
    }

    return values;
}

NonlinearFactorGraph TendonRobotModel::build_graph() const
{
    NonlinearFactorGraph graph = rod_->build_graph();

    // Base frame prior constraint
    graph.add(PriorFactor<Pose3>(rod_->get_pose_key(0), base_pose_mean_, base_pose_noise_));

    // Priors for discs (using disc indices), start at 1, no force at base disc
    for (size_t disc_idx = 1; disc_idx < num_discs_; ++disc_idx) {
        int pose_idx = tendon_config_.disc_pose_idx[disc_idx];
        int pose_idx_prev = tendon_config_.disc_pose_idx[disc_idx - 1];
        const std::vector<Vector3>& holes_prev = tendon_config_.hole_locations[disc_idx - 1];
        const std::vector<Vector3>& holes = tendon_config_.hole_locations[disc_idx];

        // Only include the external-wrench key if this disc is actually a
        // true external-wrench location; otherwise the disc's rod wrench is
        // fully determined by tendon force alone.
        std::optional<Key> ext_key = is_external_wrench_node_[pose_idx]
            ? std::optional<Key>(get_disc_wrench_key(disc_idx)) : std::nullopt;

        // Tip disc (last one): no next disc.
        bool is_tip = (disc_idx == num_discs_ - 1);
        std::optional<Key> pose_next_key = is_tip
            ? std::nullopt : std::optional<Key>(rod_->get_pose_key(tendon_config_.disc_pose_idx[disc_idx + 1]));
        const std::vector<Vector3>& holes_next = is_tip
            ? holes /* unused when is_tip */ : tendon_config_.hole_locations[disc_idx + 1];

        graph.add(TendonDiscWrenchFactor(
            rod_->get_pose_key(pose_idx_prev),
            rod_->get_pose_key(pose_idx),
            pose_next_key,
            rod_->get_wrench_key(pose_idx),
            get_tensions_key(),
            ext_key,
            holes_prev,
            holes,
            holes_next,
            stress_noise_));
    }

    // Factor that constrains the total tendon length given a displacement prior
    std::vector<Key> disc_pose_keys;
    disc_pose_keys.reserve(num_discs_);
    for (int disc_idx = 0; disc_idx < num_discs_; ++disc_idx)
        disc_pose_keys.push_back(rod_->get_pose_key(tendon_config_.disc_pose_idx[disc_idx]));

    graph.add(TendonDisplacementFactor(
        disc_pose_keys,
        get_tensions_key(),
        get_displacements_key(),
        tendon_config_.hole_locations,
        reference_lengths_,
        tendon_stiffness_,
        displacement_constraint_noise_));

    return graph;
}

void TendonRobotModel::get_J_pose_tensions(const Marginals& marginals, TendonRobotMarginals& out) const {
    // Get joint marginal between tip pose and tensions
    Key Q = get_tensions_key();
    Key T = rod_->get_pose_key(-1);
    JointMarginal joint = marginals.jointMarginalCovariance({Q, T});

    Matrix sigma_TQ = joint(T, Q);      // 6 x num_tendons
    Matrix sigma_QQ_inv = marginals.marginalInformation(Q);  // num_tendons x num_tendons

    out.J_pose_tensions = sigma_TQ * sigma_QQ_inv;
}

void TendonRobotModel::get_J_pose_displacements(const Marginals& marginals, TendonRobotMarginals& out) const {
    Key X = get_displacements_key();
    Key T = rod_->get_pose_key(-1);
    JointMarginal joint = marginals.jointMarginalCovariance({X, T});

    Matrix sigma_TX = joint(T, X);      // 6 x num_tendons
    Matrix sigma_XX_inv = marginals.marginalInformation(X);  // num_tendons x num_tendons

    out.J_pose_displacements = sigma_TX * sigma_XX_inv;
}

void TendonRobotModel::get_J_tension_displacements(const Marginals& marginals, TendonRobotMarginals& out) const {
    Key Q = get_tensions_key();
    Key X = get_displacements_key();
    JointMarginal joint = marginals.jointMarginalCovariance({Q, X});

    Matrix sigma_QX = joint(Q, X);      // num_tendons x num_tendons
    Matrix sigma_XX_inv = marginals.marginalInformation(X);  // num_tendons x num_tendons

    out.J_tension_displacements = sigma_QX * sigma_XX_inv;
}

TendonRobotMarginals TendonRobotModel::get_marginals(
    const Values& values,
    const Marginals& marginals) const
{
    TendonRobotMarginals m;

    m.rod = rod_->get_marginals(values, marginals);
    m.tendon_config = tendon_config_;

    m.tensions.mean = values.at<Vector>(get_tensions_key());
    m.tensions.cov = marginals.marginalCovariance(get_tensions_key());

    m.displacements.mean = values.at<Vector>(get_displacements_key());
    m.displacements.cov = marginals.marginalCovariance(get_displacements_key());

    m.external_wrenches.resize(num_nodes_);
    for (int i = 0; i < num_nodes_; i++) {
        std::optional<Key> key = get_external_wrench_key(i);
        if (!key) continue;

        Vector6Gaussian wrench;
        wrench.mean = values.at<Vector6>(*key);
        wrench.cov = marginals.marginalCovariance(*key);
        m.external_wrenches[i] = wrench;
    }

    // All the Jacobians we care about 
    get_J_pose_tensions(marginals, m);
    get_J_pose_displacements(marginals, m);
    get_J_tension_displacements(marginals, m);

    return m;
}
