#pragma once

#include <concepts>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>

// A BENDIER model manages its own physics factor graph and exposes symbols to the outside world for additional priors or constraints.
// A conforming model type must satisfy the BendierModel concept:
template <typename T>
concept BendierModel = requires(
    const T& m,
    const gtsam::Values& vals,
    const gtsam::Marginals& marg)
{
    typename T::ModelMarginals;
    { m.build_graph() }             -> std::same_as<gtsam::NonlinearFactorGraph>;
    { m.get_initial_values() }      -> std::same_as<gtsam::Values>;
    { m.get_marginals(vals, marg) } -> std::same_as<typename T::ModelMarginals>;
};
