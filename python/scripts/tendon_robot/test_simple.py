import numpy as np

import bendier
from bendier.visualization import TendonRobotPlotter, FramePacer

from config import get_config


def main():
    solver = bendier.TendonRobotSolver(get_config())

    sim_time = 20.0
    frame_rate = 30.0
    dt = 1.0 / frame_rate
    pacer = FramePacer(dt)

    plotter = TendonRobotPlotter()

    tensions_cov = (1e-2) ** 2 * np.eye(4)
    tip_wrench_cov = (1e-3) ** 2 * np.eye(6)

    for t in np.arange(0, sim_time, dt):
        tensions_mean = np.zeros(4)
        tensions_mean[0] = 4.5 + 4.5 * np.sin(1.2 * t)

        tip_wrench_mean = np.zeros(6)
        tip_wrench_mean[5] = 0.12 * np.sin(1.0 * t)

        tensions = bendier.VectorXGaussian(tensions_mean, tensions_cov)
        tip_wrench = bendier.Vector6Gaussian(tip_wrench_mean, tip_wrench_cov)

        solution = solver.solve(tensions, tip_wrench, None)
        plotter.update(solution)
        pacer.tick()

    plotter.close()


if __name__ == "__main__":
    main()
