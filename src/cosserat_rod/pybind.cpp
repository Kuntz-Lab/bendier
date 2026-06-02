#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "CosseratRodModel.h"
#include "CosseratRodSolver.h"

namespace py = pybind11;


void bind_cosserat_rod(py::module& m) {
    py::class_<CosseratRodState>(m, "CosseratRodState")
        .def(py::init<>())
        .def_readwrite("pose", &CosseratRodState::pose)
        .def_readwrite("stress", &CosseratRodState::stress)
        .def_readwrite("wrench", &CosseratRodState::wrench);
    
    py::class_<CosseratRodMarginals>(m, "CosseratRodMarginals")
        .def(py::init<>())
        .def_readwrite("states", &CosseratRodMarginals::states);
    
    py::class_<Solution<CosseratRodMarginals>>(m, "CosseratRodSolution")
        .def(py::init<>())
        .def_readwrite("meta", &Solution<CosseratRodMarginals>::meta)
        .def_readwrite("marginals", &Solution<CosseratRodMarginals>::marginals);

    py::class_<CosseratRodSolverConfig>(m, "CosseratRodSolverConfig")
        .def(py::init<>())  // default constructor
        .def_readwrite("base", &CosseratRodSolverConfig::base)
        .def_readwrite("rod_length", &CosseratRodSolverConfig::rod_length)
        .def_readwrite("num_nodes", &CosseratRodSolverConfig::num_nodes)
        .def_readwrite("num_magnus_terms", &CosseratRodSolverConfig::num_magnus_terms)
        .def_readwrite("K_inv", &CosseratRodSolverConfig::K_inv)
        .def_readwrite("sigma_strain_pos", &CosseratRodSolverConfig::sigma_strain_pos)
        .def_readwrite("sigma_strain_rot", &CosseratRodSolverConfig::sigma_strain_rot)
        .def_readwrite("sigma_small_force", &CosseratRodSolverConfig::sigma_small_force)
        .def_readwrite("sigma_small_moment", &CosseratRodSolverConfig::sigma_small_moment)
        .def_readwrite("sigma_base_pose_pos", &CosseratRodSolverConfig::sigma_base_pose_pos)
        .def_readwrite("sigma_base_pose_rot", &CosseratRodSolverConfig::sigma_base_pose_rot);

    py::class_<CosseratRodSolver>(m, "CosseratRodSolver")
        .def(py::init<const CosseratRodSolverConfig&>())
        .def("solve", &CosseratRodSolver::solve,
            py::arg("tip_force"),
            py::arg("tip_pose"),
            py::arg("nominal_strain"),
            py::call_guard<py::gil_scoped_release>());
}