#include "bindings.h"
#include "pybind/BindHelpers.h"

#include "utils/SolverBase.h"
#include "utils/Gaussians.h"

namespace py = pybind11;

void bind_utils(py::module& m) {
    py::class_<SolverBaseConfig>(m, "SolverBaseConfig")
        .def(py::init<>())
        .BIND_FIELD(SolverBaseConfig, linear_solver_type)
        .BIND_FIELD(SolverBaseConfig, optimizer_type)
        .BIND_FIELD(SolverBaseConfig, delta_initial)
        .BIND_FIELD(SolverBaseConfig, max_iterations);

    py::class_<SolutionMetadata>(m, "SolutionMetadata")
        .def(py::init<>())
        .BIND_FIELD(SolutionMetadata, total_time_ms)
        .BIND_FIELD(SolutionMetadata, build_time_ms)
        .BIND_FIELD(SolutionMetadata, optimize_time_ms)
        .BIND_FIELD(SolutionMetadata, marginalize_time_ms)
        .BIND_FIELD(SolutionMetadata, extract_time_ms)
        .BIND_FIELD(SolutionMetadata, iterations)
        .BIND_FIELD(SolutionMetadata, error);

    bind_gaussian<Vector6Gaussian, gtsam::Vector6, gtsam::Matrix6>(m, "Vector6Gaussian");
    bind_gaussian<Pose3Gaussian,   gtsam::Matrix4, gtsam::Matrix6>(m, "Pose3Gaussian");
    bind_gaussian<Vector3Gaussian, gtsam::Vector3, gtsam::Matrix3>(m, "Vector3Gaussian");
    bind_gaussian<VectorXGaussian, gtsam::Vector, gtsam::Matrix>(m, "VectorXGaussian");
}

PYBIND11_MODULE(_bendier, m) {
    bind_utils(m);
    bind_cosserat_rod(m);
    bind_parallel_robot(m);
    bind_tendon_robot(m);
}
