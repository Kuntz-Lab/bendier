#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "RigidRobotModel.h"
#include "RigidRobotSolver.h"
#include "pybind/BindHelpers.h"

namespace py = pybind11;

void bind_rigid_robot(py::module& m) {
    py::enum_<JointType>(m, "JointType")
        .value("REVOLUTE", JointType::Revolute)
        .value("PRISMATIC", JointType::Prismatic)
        .export_values();

    py::class_<RigidJointSpec>(m, "RigidJointSpec")
        .def(py::init<>())
        .def(py::init<Pose3Gaussian, gtsam::Vector3, JointType>(),
            py::arg("offset_calibration"),
            py::arg("axis") = gtsam::Vector3::UnitZ(),
            py::arg("type") = JointType::Revolute)
        .BIND_FIELD(RigidJointSpec, offset_calibration)
        .BIND_FIELD(RigidJointSpec, axis)
        .BIND_FIELD(RigidJointSpec, type);

    py::class_<RigidLinkState>(m, "RigidLinkState")
        .def(py::init<>())
        .BIND_FIELD(RigidLinkState, pose);

    py::class_<RigidRobotMarginals>(m, "RigidRobotMarginals")
        .def(py::init<>())
        .BIND_FIELD(RigidRobotMarginals, links)
        .BIND_FIELD(RigidRobotMarginals, offsets)
        .BIND_FIELD(RigidRobotMarginals, joints)
        .BIND_FIELD(RigidRobotMarginals, tip_pose)
        .BIND_FIELD(RigidRobotMarginals, J_tip_joints)
        .BIND_FIELD(RigidRobotMarginals, tip_wrench)
        .BIND_FIELD(RigidRobotMarginals, joint_torques);

    bind_solution<RigidRobotMarginals>(m, "RigidRobotSolution");

    py::class_<RigidRobotSolverConfig>(m, "RigidRobotSolverConfig")
        .def(py::init<
                std::vector<RigidJointSpec>,
                Pose3Gaussian,
                Pose3Gaussian,
                double, double,
                bool,
                const SolverBaseConfig&>(),
            py::arg("joints"),
            py::arg("base_pose_calibration"),
            py::arg("tip_offset_calibration"),
            py::arg("sigma_chain_rot") = 1e-6,
            py::arg("sigma_chain_pos") = 1e-6,
            py::arg("enable_wrench_sensing") = false,
            py::arg("base") = SolverBaseConfig{});

    py::class_<RigidRobotSolver>(m, "RigidRobotSolver")
        .def(py::init<const RigidRobotSolverConfig&>())
        .def("solve", &RigidRobotSolver::solve,
            py::arg("joint_prior"),
            py::arg("tip_wrench_prior") = py::none(),
            py::arg("joint_torque_meas") = py::none(),
            py::arg("tip_pose_prior") = py::none(),
            py::call_guard<py::gil_scoped_release>());
}
