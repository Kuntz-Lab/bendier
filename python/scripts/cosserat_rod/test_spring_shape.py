import numpy as np

import bendier
from bendier.visualization import CosseratRodPlotter, FramePacer
from config import get_config


def get_tip_wrench_prior(t):
    force = np.array([
        0.5 * np.cos(1.5 * t),
        0.5 * np.sin(1.0 * t),
        0.5 * np.sin(1.2 * t)
    ])

    tip_wrench_mean = np.hstack((np.zeros(3), force))

    tip_wrench_cov = 1e-6 * np.eye(6)

    sigma_amplitude = 0.01
    sigma = 1e-3 + sigma_amplitude - sigma_amplitude * np.cos(0.75 * t)
    tip_wrench_cov[3:,3:] = sigma ** 2 * np.eye(3)

    return bendier.Vector6Gaussian(tip_wrench_mean, tip_wrench_cov), None


def main():
    np.random.seed(42)

    config = get_config(rod_length=2, num_nodes=50)

    solver = bendier.CosseratRodSolver(config)

    plotter = CosseratRodPlotter(
        plot_base_plate=True,
        base_plate_size=0.05,
        plot_wrenches=True,
        force_scale=0.2,
    )

    frame_rate = 30.0
    dt = 1.0 / frame_rate
    t_final = 20.0
    num_steps = int(t_final / dt)
    pacer = FramePacer(dt)

    nominal_strain = np.zeros(6)
    nominal_strain[5] = 1.0
    nominal_strain[3] = 0.2
    nominal_strain[0] = 20.0

    for step in range(num_steps + 1):
        t = step * dt

        solution = solver.solve(*get_tip_wrench_prior(t), nominal_strain)
        plotter.update(solution)
        pacer.tick()

        progress = 100.0 * step / num_steps
        print(f"Progress: {progress:5.1f}%", end="\r")

    plotter.close()


if __name__ == "__main__":
    main()
