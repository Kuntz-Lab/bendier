"""Numerical check of TendonRobotMarginals.J_pose_tensions.

Same rationale as rigid_robot/test_jacobian.py: J_pose_tensions is derived
from the solved factor graph's marginal covariances (Sigma_TQ * Sigma_QQ^-1),
not a hand-written analytic Jacobian, so this validates it against finite
differences of the tip pose as a function of the commanded tensions.
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(__file__))

import bendier
from config import get_config

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


def solve_at(config, tensions, tight_cov, tip_wrench):
    solver = bendier.TendonRobotSolver(config)  # fresh instance -- no warm-start bias
    return solver.solve(bendier.VectorXGaussian(tensions, tight_cov), tip_wrench, None)


def test_J_pose_tensions():
    config = get_config()
    num_tendons = 4

    tensions0 = np.array([1.0, 0.5, 0.3, 0.2])
    tight_cov = (1e-8 ** 2) * np.eye(num_tendons)
    # Small, fixed tip wrench prior -- required by TendonRobotSolver, held
    # constant across all perturbations so it doesn't confound the Jacobian.
    tip_wrench = bendier.Vector6Gaussian(np.zeros(6), (1e-6 ** 2) * np.eye(6))

    sol0 = solve_at(config, tensions0, tight_cov, tip_wrench)
    J = sol0.marginals.J_pose_tensions
    assert J.shape == (6, num_tendons), f"unexpected J_pose_tensions shape {J.shape}"
    tip0 = sol0.marginals.rod.states[-1].pose.mean

    J_numeric = np.zeros((6, num_tendons))
    for i in range(num_tendons):
        t_pert = tensions0.copy()
        t_pert[i] += EPS
        tip_pert = solve_at(config, t_pert, tight_cov, tip_wrench).marginals.rod.states[-1].pose.mean
        J_numeric[:, i] = tangent_delta(tip0, tip_pert) / EPS

    err = np.max(np.abs(J - J_numeric))
    print("J_pose_tensions (marginal-derived):\n", J)
    print("J_pose_tensions (numerical FK):\n", J_numeric)
    print("max abs error:", err)
    assert err < TOLERANCE, f"J_pose_tensions does not match numerical FK Jacobian (max err {err})"
    print("PASSED: J_pose_tensions matches numerical FK Jacobian")


if __name__ == "__main__":
    test_J_pose_tensions()
