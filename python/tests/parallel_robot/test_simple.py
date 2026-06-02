import numpy as np

import bendier
from bendier.plotting.parallel_robot_plotter import ParallelRobotPlotter

from .config import get_base_config, platform_z_offset


def main():
    np.random.seed(42)

    config = get_base_config()
    solver = bendier.ParallelRobotSolver(config)

    plotter = ParallelRobotPlotter(
        plot_rod_wrenches=False,
        plot_tip_force=True,
        platform_z_offset=platform_z_offset,
        camera_azimuth=-60,
        camera_distance=2.7,
        camera_focal_point=np.array([0, 0, 0.5]),
    )

    wrench_cov = 1.0e-6 * np.eye(6)
    rod_lengths_sigma = 0.001

    sim_time = 30.0
    frame_rate = 30.0
    dt = 1.0 / frame_rate
    
    for t in np.arange(0, sim_time, dt):
        rod_lengths = 0.6 * np.ones(6) + np.sin(0.1 * t + np.arange(6)) * 0.05

        wrench_mean = np.zeros(6)
        wrench_mean[3] = 1.0 * np.sin(0.5 * t)


        solution = solver.solve(
            rod_lengths,
            rod_lengths_sigma,
            bendier.Vector6Gaussian(wrench_mean, wrench_cov),
            None,
        )

        plotter.update(solution)

        progress = 100.0 * t / max(1, sim_time - dt)
        print(f"Progress: {progress:5.1f}%", end="\r")


if __name__ == "__main__":
    main()
