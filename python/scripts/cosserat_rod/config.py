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


def _default_base():
    base = bendier.SolverBaseConfig()
    base.linear_solver_type = "MULTIFRONTAL_QR"
    base.delta_initial = 1.0
    return base


DEFAULTS = dict(
    rod_length=0.5,
    num_nodes=15,
    K_inv=get_K_inv(),
    sigma_strain_pos=0.003,
    sigma_strain_rot=0.03,
    sigma_small_force=1.0e-3,
    sigma_small_moment=1.0e-4,
    sigma_base_pose_pos=1.0e-4,
    sigma_base_pose_rot=1.0e-3,
)


def get_config(**overrides):
    params = {**DEFAULTS, **overrides}
    if "base" not in params:
        params["base"] = _default_base()
    return bendier.CosseratRodSolverConfig(**params)
