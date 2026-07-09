import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

import bendier
from bendier.viser_plotting import ViserParallelRobotPlotter
from bendier.plotting.utils import setup_plt

from config import platform_z_offset, get_config, DEFAULTS
from benchmark import ParallelRobotBaseline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tendon_robot"))
from utils import GaussianProcessNoiseModel


def get_goal_pose(t):
    wt = 2 * np.pi * (0.1) * t

    p_xy = 0.2 / 30.0 * t * np.array([np.cos(wt), np.sin(wt)])
    p_z = 0.55 + 0.15 * np.sin(2 * np.pi * (0.333) * t)
    p = np.hstack([p_xy, p_z])

    r_xy = np.radians(30) / 30.0 * t * np.array([-np.sin(wt), np.cos(wt)])
    r = np.hstack([r_xy, 0])
    R = Rotation.from_rotvec(r).as_matrix()

    return p, R


def get_tip_position_baseline(solution):
    pose_ends = np.array([rod['pose'][-1] for rod in solution])  # num_rods, 4, 4
    p_ends = pose_ends[:,:3,3]
    R_ends = pose_ends[:,:3,:3]
        
    z_offset = R_ends[:,:3,2] * platform_z_offset

    return np.mean(p_ends - z_offset, axis=0)
    

def run_sim(
        sim_time=30.0, 
        frame_rate=30.0, 
        rod_lengths_sigma=0.001,
        small_rod_lengths_sigma=1e-5,
        small_wrench_sigma=1e-3,
        actuator_f_meas_sigma=0.1,
        tip_force_prior_sigma=0.5,
        plot=True, 
        do_baseline=True):

    config = get_config()

    # Simulator to generate actuator forces on rods
    solver_sim = bendier.ParallelRobotSolver(config)
    baseline = ParallelRobotBaseline(DEFAULTS['K_inv'], DEFAULTS['base_end_poses'], DEFAULTS['tip_end_poses'], plot=False)

    # Prior solves the robot with no measurements but with big force prior
    solver_prior = bendier.ParallelRobotSolver(config)
    plotter_prior = ViserParallelRobotPlotter(
        port=8080,
        plot_rod_wrenches=False,
        plot_tip_force=False,
        platform_z_offset=platform_z_offset,
    )

    # The actual solver that solves given measurements
    solver_post = bendier.ParallelRobotSolver(config)
    plotter_post = ViserParallelRobotPlotter(
        port=8081,
        plot_rod_wrenches=False,
        plot_tip_force=True,
        platform_z_offset=platform_z_offset,
    )

    # Seperate solver just for getting jacobian, dont want to mess up warm starts
    solver_jac = bendier.ParallelRobotSolver(config)

    dt = 1.0 / frame_rate
    t = np.arange(0, sim_time, dt)
    rod_lengths_cmd = 0.6 * np.ones(6)
    
    small_wrench_cov = small_wrench_sigma ** 2 * np.eye(6)
    wrench_prior_cov = small_wrench_cov.copy()
    f_prior_cov = tip_force_prior_sigma ** 2 * np.eye(3)
    wrench_prior_cov[3:,3:] = f_prior_cov
    wrench_prior = bendier.Vector6Gaussian(np.zeros(6), wrench_prior_cov)

    rod_lengths_noise_model = GaussianProcessNoiseModel(6, frame_rate, sim_time)
    f_meas_noise_model = GaussianProcessNoiseModel(6, frame_rate, sim_time)
    f_noise_model = GaussianProcessNoiseModel(3, frame_rate, sim_time)

    data = {
        'f_gt': [], 'f_mean': [], 'f_std': [], 'f_std_prior': [],
        'p_goal': [], 'p_gt': [], 'p_mean': [], 'p_std': [], 'p_std_prior': [], 'p_baseline': []
    }

    for ti in t:
        # Solve using our method and capture uncertainty
        f_gt = f_noise_model.step(f_prior_cov)

        wrench_gt = np.zeros(6)
        wrench_gt[3:] = f_gt
        rod_lengths_gt = rod_lengths_cmd + rod_lengths_noise_model.step(rod_lengths_sigma ** 2 * np.eye(6))
        solution_sim = solver_sim.solve(
            rod_lengths_gt, 
            small_rod_lengths_sigma, 
            bendier.Vector6Gaussian(wrench_gt, small_wrench_cov), 
            None
        )
        p_gt = solution_sim.marginals.platform_pose.mean[:3,3]

        # Sample base actuator z forces
        f_meas = []
        for rod in solution_sim.marginals.rods:
            f_meas.append(rod.states[0].wrench.mean[5]) # z force on base of rod
        f_meas = np.array(f_meas) + f_meas_noise_model.step(actuator_f_meas_sigma ** 2 * np.eye(6))

        # Solve prior with no measuremtns
        solution_prior = solver_prior.solve(rod_lengths_cmd, rod_lengths_sigma, wrench_prior, None)

        solution_post = solver_post.solve(
            rod_lengths_cmd, 
            rod_lengths_sigma, 
            wrench_prior,
            bendier.ActuationForceMeas(np.array(f_meas), actuator_f_meas_sigma)
        )

        # Use best guess of wrench to solve the jacobian
        wrench_post = solution_post.marginals.platform_wrench
        solution_jac = solver_jac.solve(rod_lengths_cmd, rod_lengths_sigma, wrench_post, None)

        # Compare to baseline model if requested
        if do_baseline:
            comparison = baseline.solve(rod_lengths_gt, tip_force=wrench_gt[3:], tip_moment=wrench_gt[:3])
            p_baseline = get_tip_position_baseline(comparison)
            print(f"baseline error: {np.linalg.norm(p_baseline - p_gt)}")
            data['p_baseline'].append(p_baseline)

        # Compare to current goal pose
        p = solution_post.marginals.platform_pose.mean[:3, 3]
        R = solution_post.marginals.platform_pose.mean[:3,:3]
        p_goal, R_goal = get_goal_pose(ti)
        p_error = R.T @ (p_goal - p)
        r_error = Rotation.from_matrix(R.T @ R_goal).as_rotvec()
        twist_error = np.hstack((r_error, p_error))

        # Jacobian to step toward the goal
        J = solution_jac.marginals.rod_lengths_jacobian
        d_rod_lengths = np.linalg.pinv(J) @ twist_error
        rod_lengths_cmd += d_rod_lengths

        # Collect data, plot, display
        data['f_gt'].append(f_gt)
        data['f_mean'].append(wrench_post.mean[3:])
        data['f_std'].append(np.sqrt(np.diag(wrench_post.cov[3:,3:])))
        data['f_std_prior'].append(np.sqrt(np.diag(solution_prior.marginals.platform_wrench.cov[3:,3:])))
        data['p_goal'].append(p_goal)
        data['p_gt'].append(p_gt)
        data['p_mean'].append(solution_post.marginals.platform_pose.mean[:3,3])
        data['p_std'].append(np.sqrt(np.diag(solution_post.marginals.platform_pose.cov[3:,3:])))
        data['p_std_prior'].append(np.sqrt(np.diag(solution_prior.marginals.platform_pose.cov[3:,3:])))

        if plot:
            plotter_prior.update(solution_prior)
            plotter_post.update(solution_post, tip_force_gt=f_gt)
            time.sleep(dt)

        progress = 100.0 * ti / t[-1]
        print(f"Progress: {progress:5.1f}%", end="\r")

    plotter_prior.close()
    plotter_post.close()

    return t, {k: np.asarray(v) for k, v in data.items()}


def main():
    np.random.seed(42)

    do_baseline = True
    plot = True
    t, data = run_sim(sim_time=10.0, frame_rate=30.0, do_baseline=do_baseline, plot=plot)


    color_cycle = ['r', 'g', 'b', 'c']

    setup_plt(width=3.7, height=1.2, grid=True)
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
    
    for ii, ax in enumerate(axes[:3]):
        ax.plot(t, data['f_gt'][:,ii], 'k--', label='truth')
        ax.plot(t, data['f_mean'][:,ii], color=color_cycle[ii], label='mean')
        ax.fill_between(t, 
            data['f_mean'][:,ii] - 2 * data['f_std'][:,ii],
            data['f_mean'][:,ii] + 2 * data['f_std'][:,ii], 
            alpha=0.2, color=color_cycle[ii], interpolate=True, label=r'2-$\sigma$')
        ax.set_xlim([t[0], t[-1]+1e-1])
        if ii == 0:
            ax.legend(ncol=3, columnspacing=0.2, borderpad=0.0, borderaxespad=0.2, handlelength=1.0, handletextpad=0.2)
        ax.set_xlabel('time (sec)')

    axes[0].set_ylabel(r'force (N)')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.2)

    plt.savefig("output/figures/parallel_robot_force.pdf", bbox_inches="tight")


    setup_plt(width=2, height=3.5, grid=True)
    fig, axes = plt.subplots(2, 1, sharex=True)


    axes[0].plot(t, 1000.0 * np.sqrt(np.sum(data['p_std_prior']**2, axis=1)), 'k--', label='prior')
    axes[0].plot(t, 1000.0 * np.sqrt(np.sum(data['p_std']**2, axis=1)), 'k-', label='posterior')
    axes[0].legend(title='position (mm)', frameon=False)

    axes[1].plot(t, np.sqrt(np.sum(data['f_std_prior']**2, axis=1)), 'k--', label='prior')
    axes[1].plot(t, np.sqrt(np.sum(data['f_std']**2, axis=1)), 'k-', label='posterior')
    axes[1].legend(title='force (N)', frameon=False)
    axes[1].set_xlabel('time (sec)')

    fig.align_ylabels()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35, hspace=0.2)
    
    center = (axes[0].get_position().x0 + axes[0].get_position().x1)/2
    fig.text(center, 0.97, 'Uncertainty', ha='center', va='bottom', fontsize=8)

    plt.savefig("output/figures/parallel_robot_uncertainty.pdf", bbox_inches="tight")


    if do_baseline:
        setup_plt()
        plt.figure()
        baseline_err = 1000 * np.linalg.norm(data['p_gt'] - data['p_baseline'], axis=1)
        plt.plot(t, baseline_err, label=f"baseline (mean={np.mean(baseline_err):.3f} mm)")
        plt.xlabel("time (sec)")
        plt.ylabel("baseline error (mm)")
        plt.grid(True, alpha=0.25)
        plt.legend()

        plt.tight_layout()
        plt.savefig("output/figures/parallel_robot_baseline.pdf", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()    