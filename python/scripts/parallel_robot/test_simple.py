import time

import numpy as np

import bendier
from bendier.viser_plotting import ViserParallelRobotPlotter

from config import get_config, platform_z_offset


def main():
    np.random.seed(42)

    solver = bendier.ParallelRobotSolver(get_config())

    plotter = ViserParallelRobotPlotter(
        plot_rod_wrenches=False,
        plot_tip_force=True,
        platform_z_offset=platform_z_offset,
    )

    wrench_cov = 1.0e-6 * np.eye(6)
    rod_lengths_sigma = 0.001

    sim_time = 20.0
    frame_rate = 30.0
    dt = 1.0 / frame_rate
    
    for t in np.arange(0, sim_time, dt):
        rod_lengths = 0.6 * np.ones(6) + np.sin(0.5 * t + np.arange(6)) * 0.08

        wrench_mean = np.zeros(6)
        wrench_mean[3] = 1.0 * np.sin(1.0 * t)
        wrench_mean[4] = 1.0 * np.sin(0.8 * t + 1.0)


        solution = solver.solve(
            rod_lengths,
            rod_lengths_sigma,
            bendier.Vector6Gaussian(wrench_mean, wrench_cov),
            None,
        )

        plotter.update(solution)
        time.sleep(dt)

        progress = 100.0 * t / max(1, sim_time - dt)
        print(f"Progress: {progress:5.1f}%", end="\r")

    plotter.close()


if __name__ == "__main__":
    main()
