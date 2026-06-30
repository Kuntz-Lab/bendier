#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "CosseratRodModel.h"
#include "CosseratRodSolver.h"
#include "pybind/BindHelpers.h"

namespace py = pybind11;

void bind_cosserat_rod(py::module& m) {
    py::class_<CosseratRodState>(m, "CosseratRodState")
        .def(py::init<>())
        .BIND_FIELD(CosseratRodState, pose)
        .BIND_FIELD(CosseratRodState, stress)
        .BIND_FIELD(CosseratRodState, wrench);

    py::class_<CosseratRodMarginals>(m, "CosseratRodMarginals")
        .def(py::init<>())
        .BIND_FIELD(CosseratRodMarginals, states);

    bind_solution<CosseratRodMarginals>(m, "CosseratRodSolution");

    py::class_<CosseratRodSolverConfig>(m, "CosseratRodSolverConfig")
        .def(py::init<
                double,
                int,
                int,
                const gtsam::Matrix6&,
                double, double,
                double, double,
                double, double,
                const SolverBaseConfig&>(),
            py::arg("rod_length"),
            py::arg("num_nodes"),
            py::arg("num_magnus_terms"),
            py::arg("K_inv"),
            py::arg("sigma_strain_rot"),
            py::arg("sigma_strain_pos"),
            py::arg("sigma_small_force"),
            py::arg("sigma_small_moment"),
            py::arg("sigma_base_pose_pos"),
            py::arg("sigma_base_pose_rot"),
            py::arg("base") = SolverBaseConfig{});

    py::class_<CosseratRodSolver>(m, "CosseratRodSolver")
        .def(py::init<const CosseratRodSolverConfig&>())
        .def("solve", &CosseratRodSolver::solve,
            py::arg("tip_wrench")      = py::none(),
            py::arg("tip_pose")        = py::none(),
            py::arg("nominal_strain")  = py::none(),
            py::call_guard<py::gil_scoped_release>());
}
