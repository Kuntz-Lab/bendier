#pragma once

#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <chrono>
#include <memory>
#include <string>
#include <utility>

#include "utils/ModelConcept.h"

struct SolutionMetadata {
    double total_time_ms = 0;
    double build_time_ms = 0;
    double optimize_time_ms = 0;
    double marginalize_time_ms = 0;
    double extract_time_ms = 0;
    int iterations = 0;
    double error = 0;
};

template<typename MarginalType>
struct Solution {
    MarginalType marginals;
    SolutionMetadata meta;
};

struct SolverBaseConfig {
    std::string linear_solver_type = "MULTIFRONTAL_QR";
    std::string optimizer_type = "DOGLEG";
    double delta_initial = 1.0;
    int max_iterations = 100;
};

template<BendierModel ModelType>
class SolverBase {
public:
    explicit SolverBase(const SolverBaseConfig& config) : config_(config) {}
    virtual ~SolverBase() = default;

protected:
    Solution<typename ModelType::ModelMarginals> run_solve(
        gtsam::NonlinearFactorGraph extra = {})
    {
        using Clock = std::chrono::high_resolution_clock;
        using Ms    = std::chrono::duration<double, std::milli>;

        if (warm_start_.empty())
            warm_start_ = model_->get_initial_values();

        auto t_build = Clock::now();
        auto graph = model_->build_graph();
        graph.add(extra);
        double build_ms = Ms(Clock::now() - t_build).count();

        SolutionMetadata meta;
        auto t_opt = Clock::now();
        if (config_.optimizer_type == "GAUSS_NEWTON") {
            gtsam::GaussNewtonParams params;
            params.setLinearSolverType(config_.linear_solver_type);
            params.maxIterations = config_.max_iterations;
            gtsam::GaussNewtonOptimizer opt(graph, warm_start_, params);
            warm_start_ = opt.optimize();
            meta.error      = opt.error();
            meta.iterations = opt.iterations();
        } else {
            gtsam::DoglegParams params;
            params.setLinearSolverType(config_.linear_solver_type);
            params.deltaInitial  = config_.delta_initial;
            params.maxIterations = config_.max_iterations;
            gtsam::DoglegOptimizer opt(graph, warm_start_, params);
            warm_start_ = opt.optimize();
            meta.error      = opt.error();
            meta.iterations = opt.iterations();
        }
        meta.optimize_time_ms = Ms(Clock::now() - t_opt).count();

        auto t_marg = Clock::now();
        gtsam::Marginals marginals(graph, warm_start_);
        meta.marginalize_time_ms = Ms(Clock::now() - t_marg).count();

        meta.build_time_ms = build_ms;

        auto t_extract = Clock::now();
        auto result = model_->get_marginals(warm_start_, marginals);
        meta.extract_time_ms = Ms(Clock::now() - t_extract).count();

        meta.total_time_ms = meta.build_time_ms + meta.optimize_time_ms
                           + meta.marginalize_time_ms + meta.extract_time_ms;
        return { std::move(result), meta };
    }

    std::unique_ptr<ModelType> model_;
    SolverBaseConfig config_;
    gtsam::Values warm_start_;
};
