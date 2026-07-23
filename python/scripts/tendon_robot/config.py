import numpy as np

from bendier import TendonRobotSolverConfig, TendonInput, RoutingAngleFunction, RoutingFunctionParams


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
    tendon_input = TendonInput()

    tendon_input.routing_radius = 0.01

    tendon_input.functions = [
        RoutingAngleFunction.LINEAR,
        RoutingAngleFunction.CONSTANT,
        RoutingAngleFunction.CONSTANT,
        RoutingAngleFunction.CONSTANT
    ]

    tendon_input.params = [
        RoutingFunctionParams(angle_offset=0.0,           total_angle=2 * np.pi),
        RoutingFunctionParams(angle_offset=np.pi,         total_angle=0.0),
        RoutingFunctionParams(angle_offset=3 * np.pi / 2, total_angle=0.0),
        RoutingFunctionParams(angle_offset=0.0,           total_angle=0.0)
    ]

    return tendon_input


def get_dexterous_tendon_input():
    """Four tendons, evenly spaced 90 degrees apart, each spiraling a full
    turn (2*pi) with alternating handedness (0/180 wind one way, 90/270 the
    other) -- a symmetric "cross-wound" pattern. Used only by the
    interactive app (python/scripts/tendon_robot/app.py), not the batch
    sims/tests, which are tuned against get_tendon_input()'s routing.

    Empirically compared against get_tendon_input() and several other
    candidates (plain 90-degree-spaced straight tendons; same-handed
    helices; mixed straight+helix combinations) by simulating a full
    null-space drag (same mechanism as the app's null-space slider: SVD of
    the position Jacobian for the null direction, one damped IK correction
    per step to hold the tip in place) and measuring both how evenly the 4
    tendons participate in the null vector and how much the rod's shape
    visibly changes along the way:

    - get_tendon_input()'s routing (3 unevenly-spaced straight tendons plus
      one full-helix) leaves the null space dominated by whichever 2
      tendons happen to land opposite each other (~0.71/0.71 vs ~0.02/0.005
      magnitude) and produces very little visible shape change (~0.06mm
      over a drag that saturates a tendon at the plain-straight config's
      limit; ~0.26mm even pushed 5x further).
    - Plain 90-degree-spaced straight tendons balance the null vector
      perfectly (0.5/0.5/0.5/0.5) but the redundant direction is nearly
      pure axial co-contraction, which barely deforms this rod's shape at
      all (~0.000mm) until a tendon saturates at its tension limit --  at
      which point the "drama" is really just the position hold failing
      (3.8mm of *uncorrected tip drift*, not a clean self-motion).
    - This alternating-handed full-helix routing balances the null vector
      (~0.56/0.40/0.60/0.40) and produces real, clean self-motion: ~2.6-3x
      more visible shape change than the original routing at matched drag
      distance, with the tip held to within 1e-4 mm even pushed 5x further
      than the original routing's tested range (never saturates a tendon
      in that range) -- because winding two tendons one way and two the
      other couples bending and twist along the helix rather than relying
      on axial pretension, which is what makes the redundant motion
      visible instead of nearly inert.
    """
    tendon_input = TendonInput()

    tendon_input.routing_radius = 0.01
    tendon_input.functions = [RoutingAngleFunction.LINEAR] * 4
    tendon_input.params = [
        RoutingFunctionParams(angle_offset=np.pi / 2,     total_angle=np.pi),
        RoutingFunctionParams(angle_offset=np.pi / 2,     total_angle=-np.pi),
        RoutingFunctionParams(angle_offset=3 * np.pi / 2, total_angle=0.0),
        RoutingFunctionParams(angle_offset=np.pi, total_angle=3 * np.pi / 2),
    ]

    return tendon_input


DEFAULTS = dict(
    rod_length=0.25,
    num_discs=9,
    num_between_nodes=3,
    K_inv=get_K_inv(),
    sigma_strain_rot=0.1,
    sigma_strain_pos=0.01,
    sigma_small_force=1.0e-4,
    sigma_small_moment=1.0e-5,
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
