import argparse
import numpy as np
from scipy.spatial.transform import Rotation

import bendier
from bendier.plotting.cosserat_rod_plotter import CosseratRodPlotter
from .config import get_base_config


def get_tip_force_prior(t):
    xy_dir = np.array([np.cos(0.1 * t), np.sin(0.1 * t)])
    f_xy = 0.5 * xy_dir
    f_z = 3 * np.sin(0.3 * t)

    tip_wrench_mean = np.hstack((np.zeros(3), f_xy, f_z))

    tip_wrench_cov = 1e-6 * np.eye(6)

    sigma_amplitude = 0.1
    sigma = 1e-3 + sigma_amplitude - sigma_amplitude * np.cos(0.1 * t)
    tip_wrench_cov[3:,3:] = sigma ** 2 * np.eye(3)

    return bendier.Vector6Gaussian(tip_wrench_mean, tip_wrench_cov), None


def get_tip_moment_prior(t):
    xy_dir = np.array([np.cos(0.1 * t), np.sin(0.11 * t)])
    m_xy = 1.0 * xy_dir
    m_z = 2.0 * np.sin(0.12 * t)

    tip_wrench_mean = np.hstack((m_xy, m_z, np.zeros(3)))

    tip_wrench_cov = 1e-6 * np.eye(6)

    sigma_amplitude = 0.01
    sigma = 1e-3 + sigma_amplitude - sigma_amplitude * np.cos(0.1 * t)
    tip_wrench_cov[:3,:3] = sigma ** 2 * np.eye(3)

    return bendier.Vector6Gaussian(tip_wrench_mean, tip_wrench_cov), None


def get_tip_pose_prior(t):
    x = 0.15
    yz = 0.15 * np.array([np.sin(0.2 * t), np.cos(0.2 * t)])
    yz[1] += 0.25
    p = np.hstack((x, yz))

    r0 = np.array([0, np.pi / 6, 0])
    R0 = Rotation.from_rotvec(r0).as_matrix()
    dr = np.pi / 3 * np.array([np.sin(0.2 * t), np.sin(0.21 * t), np.sin(0.22 * t)])
    dR = Rotation.from_rotvec(dr).as_matrix()
    R = R0 @ dR

    tip_pose_mean = np.eye(4)
    tip_pose_mean[:3,:3] = R
    tip_pose_mean[:3,3] = p

    tip_pose_cov = 1e-6 * np.eye(6)

    return None, bendier.Pose3Gaussian(tip_pose_mean, tip_pose_cov)


PRIOR_GETTERS = {
    "force": get_tip_force_prior,
    "moment": get_tip_moment_prior,
    "pose": get_tip_pose_prior,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Cosserat rod prior demo.")
    parser.add_argument(
        "--tip-prior",
        choices=sorted(PRIOR_GETTERS.keys()),
        default="pose",
        help="Select which prior model to drive the simulation.",
    )
    return parser.parse_args()


def main(args):
    np.random.seed(42)

    prior_getter = PRIOR_GETTERS[args.tip_prior]
    
    config = get_base_config()
    
    solver = bendier.CosseratRodSolver(config)
    plotter = CosseratRodPlotter(
        plot_wrenches=True,
        plot_backbone_frames=True, 
        plot_tip_plate=args.tip_prior == "pose",
        camera_azimuth=60, 
        camera_distance=1.5, 
        camera_focal_point=np.array([0, 0, 0.25]))

    frame_rate = 30.0
    dt = 1.0 / frame_rate
    t_final = 120.0
    num_steps = int(t_final / dt)

    for step in range(num_steps + 1):
        t = step * dt

        solution = solver.solve(*prior_getter(t), None)
        plotter.update(solution)

        progress = 100.0 * step / num_steps
        print(f"Progress: {progress:5.1f}%", end="\r")

if __name__ == "__main__":
    main(parse_args())