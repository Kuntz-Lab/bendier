import numpy as np

import bendier
from bendier.visualization import TendonRobotPlotter, FramePacer

from config import get_K_inv

NUM_TENDONS = 3


def get_tendon_input():
    tendon_input = bendier.TendonRoutingInput()
    tendon_input.routing_radius = 0.01

    angles = np.linspace(0, 2 * np.pi, NUM_TENDONS, endpoint=False)

    # Tendon 0 spirals a full turn around the backbone to bend the rod; the
    # rest run straight (total_angle=0.0), evenly spaced around the disc to
    # counteract it.
    tendon_input.params = [bendier.RoutingFunctionParams(angle_offset=0.0, total_angle=2 * np.pi)] + [
        bendier.RoutingFunctionParams(angle_offset=float(angle), total_angle=0.0)
        for angle in angles[1:]
    ]

    return tendon_input


def get_config():
    return bendier.TendonRobotSolverConfig(
        rod_length=0.25,
        num_discs=9,
        num_between_nodes=3,
        K_inv=get_K_inv(),
        sigma_strain_rot=0.1,
        sigma_strain_pos=0.01,
        sigma_small_force=1.0e-4,
        sigma_small_moment=1.0e-5,
        sigma_base_pose_pos=1.0e-4,
        sigma_base_pose_rot=1.0e-3,
        tendon_input=get_tendon_input(),
    )


def main():
    solver = bendier.TendonRobotSolver(get_config())

    sim_time = 20.0
    frame_rate = 30.0
    dt = 1.0 / frame_rate
    pacer = FramePacer(dt)

    plotter = TendonRobotPlotter()

    tensions_cov = (1e-2) ** 2 * np.eye(NUM_TENDONS)
    tip_wrench_cov = (1e-3) ** 2 * np.eye(6)

    for t in np.arange(0, sim_time, dt):
        tensions_mean = np.zeros(NUM_TENDONS)
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
