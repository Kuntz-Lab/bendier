import numpy as np

import bendier
from bendier.plotting.tendon_robot_plotter import TendonRobotPlotter

from .config import get_base_config


def main():
    config = get_base_config()
    
    solver = bendier.TendonRobotSolver(config)
    plotter = TendonRobotPlotter()

    tensions_cov = (1e-2) ** 2 * np.eye(4)
    tip_wrench_cov = (1e-3) ** 2 * np.eye(6)

    sim_time = 30.0
    frame_rate = 30.0
    dt = 1.0 / frame_rate

    for t in np.arange(0, sim_time, dt):
        tensions_mean = np.zeros(4)
        tensions_mean[0] = 0.1 * t
        # tensions_mean[0] = 4.0
        
        tip_wrench_mean = np.zeros(6)
        tip_wrench_mean[5] = 0.1 * np.sin(0.1 * t)
        # tip_wrench_mean[3] = 0.2

        tensions = bendier.Vector4Gaussian(tensions_mean, tensions_cov)
        tip_wrench = bendier.Vector6Gaussian(tip_wrench_mean, tip_wrench_cov)

        solution = solver.solve(tensions, tip_wrench, None)
        plotter.update(solution)


if __name__ == "__main__":
    main()
