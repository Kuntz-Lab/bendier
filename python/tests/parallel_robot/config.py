import numpy as np

import bendier


def get_end_poses(angles, radius, z_offset):
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)
    poses = []
    for xi, yi in zip(xs, ys):
        pose = np.eye(4)
        pose[0, 3] = xi
        pose[1, 3] = yi
        pose[2, 3] = z_offset
        poses.append(pose)

    return poses


def get_base_poses():
    ang = 10
    angles = np.array(np.deg2rad([ang, 120 - ang, 120 + ang, 240 - ang, 240 + ang, -ang]))

    return get_end_poses(angles, radius=0.1, z_offset=0.0)


platform_z_offset = -0.1


def get_tip_poses():
    ang = 10
    angles = np.array(np.deg2rad([60 - ang, 60 + ang, 180 - ang, 180 + ang, 300 - ang, 300 + ang]))

    return get_end_poses(angles, radius=0.1, z_offset=platform_z_offset)


def get_base_config():
    r = 0.0015 / 2
    I = 0.25 * np.pi * r ** 4
    A = np.pi * r ** 2
    J = 2 * I
    E = 207.0e9
    G = 79.3e9
    
    K_inv = np.diag([
        1 / (E * I), 
        1 / (E * I),
        1 / (J * G),
        1 / (G * A),
        1 / (G * A),
        1 / (E * A)
    ])

    config = bendier.ParallelRobotSolverConfig()

    config.base.use_dense = False
    config.nodes_per_rod = 15
    config.K_inv = K_inv
    config.sigma_strain_pos = 0.0025
    config.sigma_strain_rot = 0.025
    config.sigma_small_force = 1.0e-3
    config.sigma_small_moment = 1.0e-3
    config.base_end_poses = get_base_poses()
    config.tip_end_poses = get_tip_poses()
    config.sigma_end_pose_pos= 1.0e-4
    config.sigma_end_pose_rot= 1.0e-3

    return config