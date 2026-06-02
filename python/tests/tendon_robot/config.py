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


def get_base_config():
    config = TendonRobotSolverConfig()

    config.base.use_dense = False
    config.base.linear_solver_type = "MULTIFRONTAL_QR"
    config.rod_length = 0.25
    config.num_discs = 9
    config.num_between_nodes = 3
    config.K_inv = get_K_inv()
    config.sigma_strain_pos = 0.01
    config.sigma_strain_rot = 0.1
    config.sigma_stress_force = 1.0e-4
    config.sigma_stress_moment = 1.0e-5
    config.sigma_base_pos = 1.0e-4
    config.sigma_base_rot = 1.0e-3
    config.tendon_input = get_tendon_input()

    return config


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

