import numpy as np

import bendier
from bendier.visualization import ParallelRobotPlotter, FramePacer

from config import get_end_poses, get_K_inv

NUM_RODS = 3

platform_z_offset = -0.1


def get_base_poses(radius=0.1):
    angles = np.linspace(0, 2 * np.pi, NUM_RODS, endpoint=False)
    return get_end_poses(angles, radius, z_offset=0.0)


def get_tip_poses(radius=0.1):
    # Offset by half a step so tip attachments don't sit directly above the
    # base attachments -- gives the platform some torsional stiffness.
    angles = np.linspace(0, 2 * np.pi, NUM_RODS, endpoint=False) + np.pi / NUM_RODS
    return get_end_poses(angles, radius, z_offset=platform_z_offset)


def get_config():
    return bendier.ParallelRobotSolverConfig(
        nodes_per_rod=15,
        K_inv=get_K_inv(),
        sigma_strain_pos=0.0025,
        sigma_strain_rot=0.025,
        sigma_small_force=1.0e-3,
        sigma_small_moment=1.0e-3,
        base_end_poses=get_base_poses(),
        tip_end_poses=get_tip_poses(),
        sigma_end_pose_pos=1.0e-4,
        sigma_end_pose_rot=1.0e-3,
    )


def main():
    np.random.seed(42)

    solver = bendier.ParallelRobotSolver(get_config())

    plotter = ParallelRobotPlotter(
        plot_rod_wrenches=False,
        plot_tip_force=True,
        platform_z_offset=platform_z_offset,
    )

    wrench_cov = 1.0e-6 * np.eye(6)
    rod_lengths_sigma = 0.001

    sim_time = 20.0
    frame_rate = 30.0
    dt = 1.0 / frame_rate
    pacer = FramePacer(dt)

    for t in np.arange(0, sim_time, dt):
        rod_lengths = 0.6 * np.ones(NUM_RODS) + np.sin(0.5 * t + np.arange(NUM_RODS)) * 0.08

        wrench_mean = np.zeros(6)
        wrench_mean[3] = 0.6 * np.cos(1.0 * t)
        wrench_mean[4] = 0.6 * np.cos(0.8 * t + 1.0)

        solution = solver.solve(
            rod_lengths,
            rod_lengths_sigma,
            bendier.Vector6Gaussian(wrench_mean, wrench_cov),
            None,
        )

        plotter.update(solution)
        pacer.tick()

        progress = 100.0 * t / max(1, sim_time - dt)
        print(f"Progress: {progress:5.1f}%", end="\r")

    plotter.close()


if __name__ == "__main__":
    main()
