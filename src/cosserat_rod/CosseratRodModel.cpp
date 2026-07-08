#include "CosseratRodModel.h"
#include "utils/ModelConcept.h"

#include "CosseratStrainFactor.h"
#include "CosseratStressFactor.h"
#include "BoundaryStressFactor.h"
#include <gtsam/base/Vector.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <stdexcept>

using namespace gtsam;

static_assert(BendierModel<CosseratRodModel>);

CosseratRodModel::CosseratRodModel(
    int num_nodes,
    const Matrix6& K_inv,
    SharedDiagonal strain_noise,
    SharedDiagonal stress_noise,
    int num_magnus_terms,
    double rod_length,
    const Vector6& nominal_strain)
:
    id_(next_id_++),
    num_nodes_(num_nodes),
    strain_noise_(strain_noise),
    stress_noise_(stress_noise),
    num_magnus_terms_(num_magnus_terms),
    rod_length_(rod_length),
    nominal_strain_(nominal_strain)
{
    K_inv_ = std::vector<Matrix6>(num_nodes - 1, K_inv);

    pose_keys_.reserve(num_nodes_);
    stress_keys_.reserve(num_nodes_);
    wrench_keys_.reserve(num_nodes_);

    for (int i = 0; i < num_nodes_; i++) {
        pose_keys_.push_back(  Symbol('T', 1000 * id_ + i));
        stress_keys_.push_back(Symbol('S', 1000 * id_ + i));
        wrench_keys_.push_back(Symbol('F', 1000 * id_ + i));
    }
}

void CosseratRodModel::set_rod_length(double rod_length) {
    rod_length_ = rod_length;
}

void CosseratRodModel::set_nominal_strain(const Vector6& nominal_strain) {
    nominal_strain_ = nominal_strain;
}

int CosseratRodModel::clamp_node_idx(int node_idx) const {
    if (node_idx == -1)
        return num_nodes_ - 1;

    if (node_idx < 0 || node_idx >= num_nodes_)
        throw std::out_of_range("CosseratRod: invalid node_idx");

    return node_idx;
}

Key CosseratRodModel::get_pose_key(int node_idx) const { return pose_keys_[clamp_node_idx(node_idx)]; }
Key CosseratRodModel::get_stress_key(int node_idx) const { return stress_keys_[clamp_node_idx(node_idx)]; }
Key CosseratRodModel::get_wrench_key(int node_idx) const { return wrench_keys_[clamp_node_idx(node_idx)]; }
const std::vector<Key>& CosseratRodModel::get_wrench_keys() const { return wrench_keys_; }
const std::vector<Key>& CosseratRodModel::get_pose_keys() const { return pose_keys_; }

Values CosseratRodModel::get_initial_values(const Pose3& base_pose_init) const
{
    Values values;
    double ds = rod_length_ / (num_nodes_ - 1);

    for (int i = 0; i < num_nodes_; ++i) {
        Vector3 p = Vector3(0, 0, ds * i);
        Pose3 pose = base_pose_init * Pose3(Rot3::Identity(), p);
        values.insert(pose_keys_[i], pose);
        values.insert(stress_keys_[i], Vector6(Vector6::Zero()));
        values.insert(wrench_keys_[i], Vector6(Vector6::Zero()));
    }

    return values;
}

NonlinearFactorGraph CosseratRodModel::build_graph() const
{
    NonlinearFactorGraph graph;

    std::vector<double> ds(num_nodes_ - 1, rod_length_ / (num_nodes_ - 1));

    // Cosserat kinematics and mechanics factors
    for (int i = 0; i + 1 < num_nodes_; ++i) {
        graph.add(CosseratStrainFactor(
            pose_keys_[i],
            pose_keys_[i + 1],
            stress_keys_[i],
            stress_keys_[i + 1],
            ds[i],
            nominal_strain_,
            K_inv_[i],
            strain_noise_,
            num_magnus_terms_));

        if (i == num_nodes_ - 2) {
            // Tip element: 4-key factor, no wrench key for the next node
            graph.add(CosseratStressFactor(
                pose_keys_[i],
                pose_keys_[i + 1],
                stress_keys_[i],
                stress_keys_[i + 1],
                stress_noise_));
        } else {
            graph.add(CosseratStressFactor(
                pose_keys_[i],
                pose_keys_[i + 1],
                stress_keys_[i],
                stress_keys_[i + 1],
                wrench_keys_[i + 1],
                stress_noise_));
        }
    }

    graph.add(BoundaryStressFactor(
        stress_keys_.back(),
        wrench_keys_.back(),
        stress_noise_,
        /* is_base = */ false));

    graph.add(BoundaryStressFactor(
        stress_keys_.front(),
        wrench_keys_.front(),
        stress_noise_,
        /* is_base = */ true));

    return graph;
}

CosseratRodMarginals CosseratRodModel::get_marginals(
    const Values& values,
    const Marginals& marginals) const
{
    CosseratRodMarginals solution;
    solution.states.resize(num_nodes_);

    for (int i = 0; i < num_nodes_; ++i) {
        solution.states[i].pose.mean = values.at<Pose3>(pose_keys_[i]).matrix();
        solution.states[i].pose.cov = marginals.marginalCovariance(pose_keys_[i]);

        solution.states[i].stress.mean = values.at<Vector6>(stress_keys_[i]);
        solution.states[i].stress.cov = marginals.marginalCovariance(stress_keys_[i]);

        solution.states[i].wrench.mean = values.at<Vector6>(wrench_keys_[i]);
        solution.states[i].wrench.cov = marginals.marginalCovariance(wrench_keys_[i]);
    }

    return solution;
}
