#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "TendonRobotModel.h"
#include "TendonRobotSolver.h"
#include "pybind/BindHelpers.h"

namespace py = pybind11;

void bind_tendon_robot(py::module& m) {
    py::enum_<RoutingAngleFunction>(m, "RoutingAngleFunction")
        .value("CONSTANT", RoutingAngleFunction::CONSTANT)
        .value("LINEAR", RoutingAngleFunction::LINEAR)
        .export_values();

    py::class_<RoutingFunctionParams>(m, "RoutingFunctionParams")
        .def(py::init<>())
        .def(py::init<double, double>(), py::arg("angle_offset"), py::arg("total_angle"))
        .BIND_FIELD(RoutingFunctionParams, angle_offset)
        .BIND_FIELD(RoutingFunctionParams, total_angle);

    py::class_<TendonInput>(m, "TendonInput")
        .def(py::init<>())
        .BIND_FIELD(TendonInput, functions)
        .BIND_FIELD(TendonInput, params)
        .BIND_FIELD(TendonInput, routing_radius);

    py::class_<TendonConfig>(m, "TendonConfig")
        .def(py::init<>())
        .BIND_FIELD(TendonConfig, num_discs)
        .BIND_FIELD(TendonConfig, num_tendons)
        .BIND_FIELD(TendonConfig, routing_radius)
        .BIND_FIELD(TendonConfig, disc_pose_idx)
        .BIND_FIELD(TendonConfig, no_disc_pose_idx)
        .BIND_FIELD(TendonConfig, hole_locations);

    py::class_<TendonRobotSolverConfig>(m, "TendonRobotSolverConfig")
        .def(py::init<
                double,
                int,
                int,
                const gtsam::Matrix6&,
                double, double,
                double, double,
                double, double,
                const TendonInput&,
                double,
                const std::vector<double>&,
                double,
                const SolverBaseConfig&>(),
            py::arg("rod_length"),
            py::arg("num_discs"),
            py::arg("num_between_nodes"),
            py::arg("K_inv"),
            py::arg("sigma_strain_rot"),
            py::arg("sigma_strain_pos"),
            py::arg("sigma_small_force"),
            py::arg("sigma_small_moment"),
            py::arg("sigma_base_pose_pos"),
            py::arg("sigma_base_pose_rot"),
            py::arg("tendon_input"),
            py::arg("sigma_displacement_constraint") = 1e-6,
            py::arg("axial_stiffness") = std::vector<double>{},
            py::arg("sigma_tension_nonneg") = 0.1,
            py::arg("base") = SolverBaseConfig{});

    py::class_<TendonRobotMarginals>(m, "TendonRobotMarginals")
        .def(py::init<>())
        .BIND_FIELD(TendonRobotMarginals, rod)
        .BIND_FIELD(TendonRobotMarginals, tendon_config)
        .BIND_FIELD(TendonRobotMarginals, external_wrenches)
        .BIND_FIELD(TendonRobotMarginals, tensions)
        .BIND_FIELD(TendonRobotMarginals, displacements)
        .BIND_FIELD(TendonRobotMarginals, J_pose_tensions)
        .BIND_FIELD(TendonRobotMarginals, J_pose_displacements)
        .BIND_FIELD(TendonRobotMarginals, J_tension_displacements);

    bind_solution<TendonRobotMarginals>(m, "TendonRobotSolution");

    py::class_<TendonRobotSolver>(m, "TendonRobotSolver")
        .def(py::init<const TendonRobotSolverConfig&>())
        .def("solve", &TendonRobotSolver::solve,
             py::arg("tensions"),
             py::arg("tip_wrench")         = py::none(),
             py::arg("tip_position_meas")  = py::none(),
             py::arg("displacement_meas")  = py::none(),
             py::call_guard<py::gil_scoped_release>());
}
