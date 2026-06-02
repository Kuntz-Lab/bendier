import numpy as np

import bendier
from bendier.plotting.cosserat_rod_plotter import CosseratRodPlotter
from .config import get_base_config


def get_tip_wrench_prior(t):
    force = np.array([
        3.0 * (-0.5 * np.cos(0.2 * t) + 0.5),
        0.1 * np.sin(0.14 * t),
        0.1 * np.sin(0.16 * t)
    ])

    tip_wrench_mean = np.hstack((np.zeros(3), force))

    tip_wrench_cov = 1e-6 * np.eye(6)

    sigma_amplitude = 0.01
    sigma = 1e-3 + sigma_amplitude - sigma_amplitude * np.cos(0.1 * t)
    tip_wrench_cov[3:,3:] = sigma ** 2 * np.eye(3)
    
    return bendier.Vector6Gaussian(tip_wrench_mean, tip_wrench_cov), None


def main():
    np.random.seed(42)

    config = get_base_config()
    config.rod_length = 2
    config.num_nodes = 50

    solver = bendier.CosseratRodSolver(config)
    plotter = CosseratRodPlotter(
        plot_base_plate=True,
        base_plate_size=0.05,
        plot_wrenches=True, 
        force_scale=0.2,
        camera_azimuth=-60, 
        camera_distance=1.0, 
        camera_focal_point=np.array([0.3, 0, 0])
    )

    frame_rate = 30.0
    dt = 1.0 / frame_rate
    t_final = 120.0
    num_steps = int(t_final / dt)

    nominal_strain = np.zeros(6)
    nominal_strain[5] = 1.0
    nominal_strain[3] = 0.2
    nominal_strain[0] = 20.0

    for step in range(num_steps + 1):
        t = step * dt

        solution = solver.solve(*get_tip_wrench_prior(t), nominal_strain)
        plotter.update(solution)

        progress = 100.0 * step / num_steps
        print(f"Progress: {progress:5.1f}%", end="\r")

if __name__ == "__main__":
    main()