#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "ParallelRobotModel.h"
#include "ParallelRobotSolver.h"
#include "pybind/BindHelpers.h"

namespace py = pybind11;

void bind_parallel_robot(py::module& m) {
    py::class_<ParallelRobotSolverConfig>(m, "ParallelRobotSolverConfig")
        .def(py::init<
                int,
                const gtsam::Matrix6&,
                double, double,
                double, double,
                std::vector<gtsam::Matrix4>,
                std::vector<gtsam::Matrix4>,
                double, double,
                const SolverBaseConfig&>(),
            py::arg("nodes_per_rod"),
            py::arg("K_inv"),
            py::arg("sigma_strain_rot"),
            py::arg("sigma_strain_pos"),
            py::arg("sigma_small_force"),
            py::arg("sigma_small_moment"),
            py::arg("base_end_poses"),
            py::arg("tip_end_poses"),
            py::arg("sigma_end_pose_pos"),
            py::arg("sigma_end_pose_rot"),
            py::arg("base") = SolverBaseConfig{});
    
    py::class_<ParallelRobotMarginals>(m, "ParallelRobotMarginals")
        .def(py::init<>())
        .BIND_FIELD(ParallelRobotMarginals, rods)
        .BIND_FIELD(ParallelRobotMarginals, platform_pose)
        .BIND_FIELD(ParallelRobotMarginals, platform_wrench)
        .BIND_FIELD(ParallelRobotMarginals, rod_lengths_jacobian)
        .BIND_FIELD(ParallelRobotMarginals, tip_wrench_jacobian);
        
    bind_solution<ParallelRobotMarginals>(m, "ParallelRobotSolution");

    py::class_<ActuationForceMeas>(m, "ActuationForceMeas")
        .def(py::init<>())
        .def(py::init<const gtsam::Vector&, double>(),
            py::arg("meas"), py::arg("sigma"))
        .BIND_FIELD(ActuationForceMeas, meas)
        .BIND_FIELD(ActuationForceMeas, sigma);

    py::class_<ParallelRobotSolver>(m, "ParallelRobotSolver")
        .def(py::init<const ParallelRobotSolverConfig&>(), py::arg("config"))
        .def("solve", &ParallelRobotSolver::solve, 
            py::arg("rod_lengths"),
            py::arg("sigma_rod_lengths"),
            py::arg("wrench"),
            py::arg("f_meas") = py::none(),
            py::call_guard<py::gil_scoped_release>());
}
