"""Numerical check of RigidRobotMarginals.J_tip_joints.

J_tip_joints is derived from the solved factor graph's marginal covariances
(Sigma_TQ * Sigma_QQ^-1, not a hand-written analytic Jacobian), so this
validates it against straightforward finite differences of the tip pose as
a function of the commanded joint values -- perturb one joint at a time
(via a very tight joint prior, effectively pinning the solve to that exact
configuration) and compare the resulting tip-pose tangent-space delta to
the column J_tip_joints predicts.
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(__file__))

import bendier
from config import load_urdf, build_joint_specs, build_base_calibration, build_tip_offset_calibration

EPS = 1e-6
TOLERANCE = 1e-3


def tangent_delta(pose0, pose1):
    """Body-frame (at pose0) twist taking pose0 to pose1 -- matches the
    (angular, linear) ordering of gtsam::Pose3::Logmap used throughout the
    C++ model, to first order (exact enough at EPS-scale perturbations)."""
    R0, p0 = pose0[:3, :3], pose0[:3, 3]
    R1, p1 = pose1[:3, :3], pose1[:3, 3]
    rot_err = Rotation.from_matrix(R0.T @ R1).as_rotvec()
    pos_err = R0.T @ (p1 - p0)
    return np.concatenate([rot_err, pos_err])


def solve_at(config, q, tight_cov):
    solver = bendier.RigidRobotSolver(config)  # fresh instance -- no warm-start bias
    return solver.solve(bendier.VectorXGaussian(q, tight_cov))


def test_J_tip_joints():
    urdf = load_urdf()
    specs = build_joint_specs(urdf)
    base_cal = build_base_calibration()
    tip_cal = build_tip_offset_calibration(urdf)
    config = bendier.RigidRobotSolverConfig(specs, base_cal, tip_cal)

    num_joints = len(specs)
    q0 = np.array([0.3, 0.5, 0.0, -1.2, 0.0, 1.0, 0.2])
    tight_cov = (1e-8 ** 2) * np.eye(num_joints)

    sol0 = solve_at(config, q0, tight_cov)
    J = sol0.marginals.J_tip_joints
    assert J.shape == (6, num_joints), f"unexpected J_tip_joints shape {J.shape}"
    tip0 = sol0.marginals.tip_pose.mean

    J_numeric = np.zeros((6, num_joints))
    for i in range(num_joints):
        q_pert = q0.copy()
        q_pert[i] += EPS
        tip_pert = solve_at(config, q_pert, tight_cov).marginals.tip_pose.mean
        J_numeric[:, i] = tangent_delta(tip0, tip_pert) / EPS

    err = np.max(np.abs(J - J_numeric))
    print("J_tip_joints (marginal-derived):\n", J)
    print("J_tip_joints (numerical FK):\n", J_numeric)
    print("max abs error:", err)
    assert err < TOLERANCE, f"J_tip_joints does not match numerical FK Jacobian (max err {err})"
    print("PASSED: J_tip_joints matches numerical FK Jacobian")


if __name__ == "__main__":
    test_J_tip_joints()
