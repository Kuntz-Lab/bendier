import numpy as np

import bendier


def get_K_inv(rod_diameter=0.003, youngs_modulus=40.0e9, shear_modulus=15.0e9):
    radius = rod_diameter / 2.0
    area = np.pi * radius**2
    moment = np.pi * radius**4 / 4.0
    polar_moment = 2.0 * moment

    k_bending = youngs_modulus * moment
    k_torsion = shear_modulus * polar_moment
    k_shear = shear_modulus * area
    k_extension = youngs_modulus * area

    return np.diag(
        1.0 / np.array([
            k_bending, 
            k_bending, 
            k_torsion, 
            k_shear, 
            k_shear, 
            k_extension
        ])
    )



def get_base_config():

    config = bendier.CosseratRodSolverConfig()

    config.base.linear_solver_type = "MULTIFRONTAL_QR"
    config.base.use_dense = False
    config.base.delta_initial = 1.0

    config.rod_length = 0.5
    config.num_nodes = 15
    config.K_inv = get_K_inv()
    config.num_magnus_terms = 4

    config.sigma_strain_pos = 0.003
    config.sigma_strain_rot = 0.03

    config.sigma_small_force = 1.0e-3
    config.sigma_small_moment = 1.0e-4

    config.sigma_base_pose_pos = 1.0e-4
    config.sigma_base_pose_rot = 1.0e-3

    return config