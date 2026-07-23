# TODO: review this file
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

# J_pose_displacements/J_tension_displacements have much larger-magnitude
# entries than J_pose_tensions (observed ~50-98 vs ~0.001-0.6, for the
# routing in get_config()), so the same *absolute* TOLERANCE that works for
# J_pose_tensions is mismatched to scale here -- compare relative error
# against the Jacobian's own scale instead.
DISPLACEMENT_RELATIVE_TOLERANCE = 1e-3


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


def consistent_displacement0(config, tensions0, tip_wrench):
    """A displacement vector picked by hand (e.g. [0.008, -0.004, 0.006,
    0.002]) isn't guaranteed to be physically consistent for a given tendon
    routing -- commanding one that isn't forces the solver to reconcile an
    unreachable/highly-strained state, which is slow to converge (184
    iterations observed for a hand-picked vector, vs ~23 below) and lands
    far enough from a true equilibrium to throw off the Jacobian comparison.
    Sidestep this entirely by deriving a guaranteed-consistent displacement
    from an ordinary, fast-converging tension-commanded solve (same tensions0
    as test_J_pose_tensions) and reading off its resulting displacement.
    """
    solver = bendier.TendonRobotSolver(config)
    tight_cov = (1e-8 ** 2) * np.eye(len(tensions0))
    sol = solver.solve(bendier.VectorXGaussian(tensions0, tight_cov), tip_wrench, None)
    return sol.marginals.displacements.mean


def solve_at_displacement(solver, displacement, tight_cov, tip_wrench):
    """Displacement-commanded solve: tension floats free (broad/
    uninformative prior) rather than being commanded -- matches
    tendon_robot/app.py's TENSION_FREE_SIGMA convention, since that's the
    actual physically-commandable quantity these two Jacobians exist to
    support (see TendonRobotModel.h's displacement_constraint_noise
    comment). Takes an existing solver (rather than building one) so
    callers can warm-start each perturbation from the previous nearby
    solution -- for a self-consistent base point (see
    consistent_displacement0) the perturbations then converge in a
    handful of iterations instead of restarting from the cold, straight
    initial guess every time.
    """
    num_tendons = len(displacement)
    loose_tensions = bendier.VectorXGaussian(np.zeros(num_tendons), (20.0 ** 2) * np.eye(num_tendons))
    displacement_meas = bendier.VectorXGaussian(displacement, tight_cov)
    return solver.solve(loose_tensions, tip_wrench, None, displacement_meas)


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


def test_J_pose_displacements():
    config = get_config()
    num_tendons = 4

    tight_cov = (1e-8 ** 2) * np.eye(num_tendons)
    tip_wrench = bendier.Vector6Gaussian(np.zeros(6), (1e-6 ** 2) * np.eye(6))

    # Away from the straight/zero-tension rest state: J_pose_displacements
    # is empirically close to singular right at all-zero displacement
    # (observed singular values [17, 13, 0.003] there during interactive
    # testing), which would make this comparison numerically unstable for
    # reasons unrelated to whether the Jacobian itself is correct. Reuse
    # tensions0 from test_J_pose_tensions to get a displacement away from
    # that singularity that's also guaranteed physically consistent.
    tensions0 = np.array([1.0, 0.5, 0.3, 0.2])
    displacement0 = consistent_displacement0(config, tensions0, tip_wrench)

    solver = bendier.TendonRobotSolver(config)  # reused/warm-started below
    sol0 = solve_at_displacement(solver, displacement0, tight_cov, tip_wrench)
    J = sol0.marginals.J_pose_displacements
    assert J.shape == (6, num_tendons), f"unexpected J_pose_displacements shape {J.shape}"
    tip0 = sol0.marginals.rod.states[-1].pose.mean

    J_numeric = np.zeros((6, num_tendons))
    for i in range(num_tendons):
        d_pert = displacement0.copy()
        d_pert[i] += EPS
        tip_pert = solve_at_displacement(
            solver, d_pert, tight_cov, tip_wrench).marginals.rod.states[-1].pose.mean
        J_numeric[:, i] = tangent_delta(tip0, tip_pert) / EPS

    err = np.max(np.abs(J - J_numeric))
    rel_err = err / np.max(np.abs(J))
    print("J_pose_displacements (marginal-derived):\n", J)
    print("J_pose_displacements (numerical FK):\n", J_numeric)
    print("max abs error:", err, " max relative error:", rel_err)
    assert rel_err < DISPLACEMENT_RELATIVE_TOLERANCE, (
        f"J_pose_displacements does not match numerical FK Jacobian (max relative err {rel_err})")
    print("PASSED: J_pose_displacements matches numerical FK Jacobian")


def test_J_tension_displacements():
    config = get_config()
    num_tendons = 4

    tight_cov = (1e-8 ** 2) * np.eye(num_tendons)
    tip_wrench = bendier.Vector6Gaussian(np.zeros(6), (1e-6 ** 2) * np.eye(6))

    tensions0 = np.array([1.0, 0.5, 0.3, 0.2])
    displacement0 = consistent_displacement0(config, tensions0, tip_wrench)

    solver = bendier.TendonRobotSolver(config)  # reused/warm-started below
    sol0 = solve_at_displacement(solver, displacement0, tight_cov, tip_wrench)
    J = sol0.marginals.J_tension_displacements
    assert J.shape == (num_tendons, num_tendons), f"unexpected J_tension_displacements shape {J.shape}"
    tension0 = sol0.marginals.tensions.mean

    J_numeric = np.zeros((num_tendons, num_tendons))
    for i in range(num_tendons):
        d_pert = displacement0.copy()
        d_pert[i] += EPS
        tension_pert = solve_at_displacement(solver, d_pert, tight_cov, tip_wrench).marginals.tensions.mean
        J_numeric[:, i] = (tension_pert - tension0) / EPS

    err = np.max(np.abs(J - J_numeric))
    rel_err = err / np.max(np.abs(J))
    print("J_tension_displacements (marginal-derived):\n", J)
    print("J_tension_displacements (numerical):\n", J_numeric)
    print("max abs error:", err, " max relative error:", rel_err)
    assert rel_err < DISPLACEMENT_RELATIVE_TOLERANCE, (
        f"J_tension_displacements does not match numerical Jacobian (max relative err {rel_err})")
    print("PASSED: J_tension_displacements matches numerical Jacobian")


if __name__ == "__main__":
    test_J_pose_tensions()
    test_J_pose_displacements()
    test_J_tension_displacements()
