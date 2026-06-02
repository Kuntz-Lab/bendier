#include "SolverBase.h"

#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <limits>

using namespace gtsam;


SolverBase::SolverBase(const SolverBaseConfig& config) 
:
    config_(config)
{}


inline auto now() {
    return std::chrono::high_resolution_clock::now();
}


template <typename ClockTimePoint>
inline double ms(const ClockTimePoint& start, const ClockTimePoint& stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}


void SolverBase::optimize_dense_benchmark(const DoglegParams& params, SolutionMetadata& meta) {
    double last_error = std::numeric_limits<double>::infinity();
    double error;
    int iterations = 0;

    // A more fair comparison might include comparing SparseMatrix ldlt to gtsam elimination
    for (int i = 0; i < params.maxIterations; i++) {
        // Linearize each factor
        auto linear = graph_.linearize(values_);
        
        // Form and solve normal equations for this iter
        auto [A, b] = linear->hessian();
        Eigen::LLT<Matrix, Eigen::Upper> llt(A);
        Vector delta = llt.solve(b);

        /*
        Note doing it this way is much slower due to the dense multiplication
        But it is perfectly possible to accumulate the hessian like above without gtsam.
        Therefore, the following would be an unfair comparison:

        auto [J, e] = linear->jacobian();
        Matrix A = J.transpose() * J;
        Vector b = J.transpose() * e;
        */

        // Apply big vector to each element in values
        VectorValues delta_values(delta, Scatter(*linear));
        values_ = values_.retract(delta_values);

        // Stopping conditions
        error = graph_.error(values_);
        double abs_error_change = last_error - error;
        double rel_error_change = abs_error_change / last_error;
        last_error = error;
        iterations++;

        if (abs_error_change < params.absoluteErrorTol || 
            rel_error_change < params.relativeErrorTol)
            break;
    }

    meta.error = error;
    meta.iterations = iterations;
}


SolutionMetadata SolverBase::optimize() {
    auto start = now();
    auto start_build = start;

    build_graph();
    
    auto stop_build = now();
    auto start_optimize = stop_build;

    DoglegParams params;
    params.setLinearSolverType(config_.linear_solver_type);
    params.deltaInitial = config_.delta_initial;

    // params.absoluteErrorTol = 1e-3;
    // params.relativeErrorTol = 1e-3;

    SolutionMetadata meta;
    // If we want to use dense solver, e.g. for comparison 
    if (config_.use_dense) {
        optimize_dense_benchmark(params, meta);
    } else {
        DoglegOptimizer optimizer(graph_, values_, params);
        values_ = optimizer.optimize();
        meta.error = optimizer.error();
        meta.iterations = optimizer.iterations();
    }
    
    auto stop_optimize = now();
    auto start_marginalize = stop_optimize;

    marginals_ = Marginals(graph_, values_);

    auto stop_marginalize = now();
    auto start_extract = stop_marginalize;

    extract_solution();

    auto stop_extract = now();
    auto stop = stop_extract;

    
    meta.total_time_ms       = ms(start, stop);
    meta.build_time_ms       = ms(start_build, stop_build);
    meta.optimize_time_ms    = ms(start_optimize, stop_optimize);
    meta.marginalize_time_ms = ms(start_marginalize, stop_marginalize);
    meta.extract_time_ms     = ms(start_extract, stop_extract);



    return meta;
}