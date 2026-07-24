import numpy as np

from bendier import TendonRobotSolverConfig, TendonRoutingInput, RoutingFunctionParams


def get_K_inv():
    rod_diameter = 0.0012
    youngs_modulus = 40.0e9
    shear_modulus = 15.0e9

    I = (np.pi * rod_diameter**4) / 64.0
    J = 2 * I
    A = (np.pi * rod_diameter**2) / 4.0

    k_bending = youngs_modulus * I
    k_torsion = shear_modulus * J
    k_shear = shear_modulus * A
    k_extension = youngs_modulus * A

    K_inv = np.eye(6)
    K_inv[0,0] = 1 / k_bending
    K_inv[1,1] = 1 / k_bending
    K_inv[2,2] = 1 / k_torsion
    K_inv[3,3] = 1 / k_shear
    K_inv[4,4] = 1 / k_shear
    K_inv[5,5] = 1 / k_extension

    return K_inv


def get_tendon_input():
    tendon_input = TendonRoutingInput()

    tendon_input.routing_radius = 0.01

    tendon_input.params = [
        RoutingFunctionParams(angle_offset=0.0,           total_angle=2 * np.pi),
        RoutingFunctionParams(angle_offset=np.pi,         total_angle=0.0),
        RoutingFunctionParams(angle_offset=3 * np.pi / 2, total_angle=0.0),
        RoutingFunctionParams(angle_offset=0.0,           total_angle=0.0)
    ]

    return tendon_input


def get_dexterous_tendon_input():
    """Five tendons, evenly spaced 72 degrees apart, each spiraling a full
    turn (2*pi) with alternating handedness -- the same "cross-wound" idea
    as the previous 4-tendon routing (winding adjacent tendons opposite ways
    couples bending and twist along the helix, rather than relying on axial
    pretension, which is what makes the extra actuation DOF produce real
    self-motion instead of nearly inert axial co-contraction -- see git
    history on this function for the 4-tendon empirical comparison that
    established that). Used only by the interactive app
    (python/scripts/tendon_robot/app.py), not the batch sims/tests, which
    are tuned against get_tendon_input()'s routing.

    Not yet re-validated at 5 tendons the way the 4-tendon version was
    (null-space balance / shape-change comparison against alternative
    routings) -- this is a straightforward generalization of the working
    4-tendon pattern's geometry, a starting point for further tuning rather
    than a re-confirmed result.
    """
    tendon_input = TendonRoutingInput()

    n = 5
    tendon_input.routing_radius = 0.01
    tendon_input.params = [
        RoutingFunctionParams(
            angle_offset=i * (2 * np.pi / n),
            total_angle=(2 * np.pi if i % 2 == 0 else -2 * np.pi))
        for i in range(n)
    ]

    return tendon_input


DEFAULTS = dict(
    rod_length=0.25,
    num_discs=9,
    num_between_nodes=3,
    K_inv=get_K_inv(),
    sigma_constitutive_rot=0.1,
    sigma_constitutive_pos=0.01,
    sigma_equilibrium_force=1.0e-4,
    sigma_equilibrium_moment=1.0e-5,
    sigma_base_pose_pos=1.0e-4,
    sigma_base_pose_rot=1.0e-3,
    tendon_input=get_tendon_input(),
)


def get_config(**overrides):
    params = {**DEFAULTS, **overrides}
    return TendonRobotSolverConfig(**params)


class SimParams:
    def __init__(self):
        self.sim_time = 30.0
        self.frame_rate = 30.0
        self.tau = 1.0

        self.f_prior_cov = 0.05**2 * np.eye(3)
        self.small_wrench_cov = 1e-5 ** 2 * np.eye(6)

        self.wrench_prior_cov = self.small_wrench_cov.copy()
        self.wrench_prior_cov[3:,3:] = self.f_prior_cov

        self.q_meas_cov = 0.1**2 * np.eye(4)
        self.p_meas_cov = 0.0001**2 * np.eye(3)
        self.small_q_cov = 0.001**2 * np.eye(4)
