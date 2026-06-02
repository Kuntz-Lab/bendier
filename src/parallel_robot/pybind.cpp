#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "ParallelRobotModel.h"
#include "ParallelRobotSolver.h"

namespace py = pybind11;


void bind_parallel_robot(py::module& m) {
    py::class_<ParallelRobotSolverConfig>(m, "ParallelRobotSolverConfig")
        .def(py::init<>())
        .def_readwrite("base", &ParallelRobotSolverConfig::base)
        .def_readwrite("nodes_per_rod", &ParallelRobotSolverConfig::nodes_per_rod)
        .def_readwrite("K_inv", &ParallelRobotSolverConfig::K_inv)
        .def_readwrite("sigma_strain_pos", &ParallelRobotSolverConfig::sigma_strain_pos)
        .def_readwrite("sigma_strain_rot", &ParallelRobotSolverConfig::sigma_strain_rot)
        .def_readwrite("sigma_small_force", &ParallelRobotSolverConfig::sigma_small_force)
        .def_readwrite("sigma_small_moment", &ParallelRobotSolverConfig::sigma_small_moment)
        .def_readwrite("base_end_poses", &ParallelRobotSolverConfig::base_end_poses)
        .def_readwrite("tip_end_poses", &ParallelRobotSolverConfig::tip_end_poses)
        .def_readwrite("sigma_end_pose_pos", &ParallelRobotSolverConfig::sigma_end_pose_pos)
        .def_readwrite("sigma_end_pose_rot", &ParallelRobotSolverConfig::sigma_end_pose_rot);
    
    py::class_<ParallelRobotMarginals>(m, "ParallelRobotMarginals")
        .def(py::init<>())
        .def_readwrite("rods", &ParallelRobotMarginals::rods)
        .def_readwrite("platform_pose", &ParallelRobotMarginals::platform_pose)
        .def_readwrite("platform_wrench", &ParallelRobotMarginals::platform_wrench)
        .def_readwrite("rod_lengths_jacobian", &ParallelRobotMarginals::rod_lengths_jacobian)
        .def_readwrite("tip_wrench_jacobian", &ParallelRobotMarginals::tip_wrench_jacobian);
        
    py::class_<Solution<ParallelRobotMarginals>>(m, "ParallelRobotSolution")
        .def(py::init<>())
        .def_readwrite("meta", &Solution<ParallelRobotMarginals>::meta)
        .def_readwrite("marginals", &Solution<ParallelRobotMarginals>::marginals);

    py::class_<ActuationForceMeas>(m, "ActuationForceMeas")
        .def(py::init<>())
        .def(py::init<const gtsam::Vector6&, double>(),
            py::arg("meas"), py::arg("sigma"))
        .def_readwrite("f_meas", &ActuationForceMeas::meas)
        .def_readwrite("cov", &ActuationForceMeas::sigma);

    py::class_<ParallelRobotSolver>(m, "ParallelRobotSolver")
        .def(py::init<const ParallelRobotSolverConfig&>(), py::arg("config"))
        .def("solve", &ParallelRobotSolver::solve, 
            py::arg("rod_lengths"),
            py::arg("sigma_rod_lengths"),
            py::arg("wrench"),
            py::arg("f_meas"),
            py::call_guard<py::gil_scoped_release>());
}