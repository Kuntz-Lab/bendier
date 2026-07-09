import numpy as np
import matplotlib.pyplot as plt

from bendier import TendonRobotSolver, Vector6Gaussian, VectorXGaussian, Vector3Gaussian

from bendier.visualization import TendonRobotPlotter, setup_plt, FramePacer
from config import get_config, SimParams, DEFAULTS
from utils import generate_waypoint_trajectory, GaussianProcessNoiseModel, cov_to_std
from benchmark import TendonRobotBaseline

    
def run_sim(params, do_baseline, plot):
    
    solver_config = get_config()

    t, positions, tensions_nominal, _ = generate_waypoint_trajectory(params.sim_time, frame_rate=params.frame_rate)

    # A solver to simulate the nominal trajectory of the robot, given open loop tensions
    solver_nominal = TendonRobotSolver(solver_config)
    plotter_nominal = TendonRobotPlotter(port=8080, plot_tip_force=True)

    # Setup baseline solver (get holes from dummy solution)
    dummy_solution = solver_nominal.solve(VectorXGaussian(np.zeros(4), np.eye(4)), Vector6Gaussian(np.zeros(6), np.eye(6)), None)
    solver_baseline = TendonRobotBaseline(K_inv=DEFAULTS['K_inv'], num_discs=DEFAULTS['num_discs'], rod_length=DEFAULTS['rod_length'], holes=dummy_solution.marginals.tendon_config.hole_locations)

    # Simulator to sample tip pose data
    solver_gt = TendonRobotSolver(solver_config)

    # Solver that does the actual inference using tip pose data
    solver_post = TendonRobotSolver(solver_config)
    plotter_post = TendonRobotPlotter(port=8081, plot_tip_force=True)

    # Use a separate solver for these so it doesn't mess up our performance metrics
    solver_jac = TendonRobotSolver(solver_config)

    pacer = FramePacer(1.0 / params.frame_rate)

    # Setup Jacobian control
    damping = 1e-2
    q_cmd = np.array([0.5, 0.1, 0.1, 0.1])
    
    # Continuous time noise models for smoothness
    q_noise_model = GaussianProcessNoiseModel(4, params.frame_rate, params.sim_time, tau=params.tau)
    p_noise_model = GaussianProcessNoiseModel(3, params.frame_rate, params.sim_time, tau=params.tau)
    f_noise_model = GaussianProcessNoiseModel(3, params.frame_rate, params.sim_time, tau=params.tau)

    # Setup data collection
    data = {
        'p_mean': [], 'p_std':[], 'p_meas': [], 'p_nominal': [], 'p_goal': [], 'p_baseline': [],
        'f_mean': [], 'f_nees': [], 'q_nees': [], 'f_std':[], 'f_gt': []
    }

    for ti, p_goal, q_nominal in zip(t, positions, tensions_nominal):
        f_gt = f_noise_model.step(params.f_prior_cov)

        # Nominal solution with no force correction, still need to add noise and re solve
        q_noise = q_noise_model.step(params.q_meas_cov)
        q_nominal_gt = q_nominal + q_noise
        solution_nominal = solver_nominal.solve(
            VectorXGaussian(q_nominal_gt, params.small_q_cov),
            Vector6Gaussian(np.hstack((np.zeros(3), f_gt)), params.small_wrench_cov),
            None
        )
        p_nominal = solution_nominal.marginals.rod.states[-1].pose.mean[:3,3]

        # Compare simulator solver to baseline solver if requested
        if do_baseline:
            solution_baseline = solver_baseline.solve(q_nominal_gt, f_gt)
            p_baseline = solution_baseline[-1]['p']
            print(f"baseline error: {np.linalg.norm(p_baseline - p_nominal)}")
            data['p_baseline'].append(p_baseline)

        # Simulated solution to sample position from, small covariances
        q_gt = q_cmd + q_noise
        solution_gt = solver_gt.solve(
            VectorXGaussian(q_gt, params.small_q_cov),
            Vector6Gaussian(np.hstack((np.zeros(3), f_gt)), params.small_wrench_cov),
            None
        )

        # Sample the position
        p_gt = solution_gt.marginals.rod.states[-1].pose.mean[:3,3]
        p_meas = p_gt + p_noise_model.step(params.p_meas_cov)

        # Use the sampled position as a prior on tip pose
        solution_post = solver_post.solve(
            VectorXGaussian(q_cmd, params.q_meas_cov),
            Vector6Gaussian(np.zeros(6), params.wrench_prior_cov),
            Vector3Gaussian(p_meas, params.p_meas_cov)
        )
        
        # Evaluate the Jacobian for control using the estimated tip wrench
        wrench_post = solution_post.marginals.external_wrenches[-1]
        solution_jac = solver_jac.solve(
            VectorXGaussian(q_cmd, params.small_q_cov),
            wrench_post,
            None
        )
        
        J_position = solution_jac.marginals.J_pose_tensions[3:]
        R_jac = solution_jac.marginals.rod.states[-1].pose.mean[:3,:3]
        p_error = R_jac.T @ (p_goal - p_meas)

        JTJ = J_position.T @ J_position
        A = JTJ + (damping**2) * np.eye(JTJ.shape[0])
        b = J_position.T @ p_error
        dq = np.linalg.solve(A, b)

        q_cmd = q_cmd + dq

        # Tensions cannot be negative
        q_cmd = np.maximum(q_cmd, np.zeros(4))

        data['p_std'].append(cov_to_std(solution_post.marginals.rod.states[-1].pose.cov[3:,3:]))
        data['p_meas'].append(p_meas)
        data['p_goal'].append(p_goal)
        data['p_nominal'].append(p_nominal)
        data['f_gt'].append(f_gt)
        data['f_mean'].append(wrench_post.mean[3:])
        data['f_std'].append(cov_to_std(wrench_post.cov[3:,3:]))

        if plot:
            plotter_post.update(solution_post, p_desired=p_goal, tip_force_gt=f_gt)
            plotter_nominal.update(solution_nominal, p_desired=p_goal, tip_force_gt=f_gt)
            pacer.tick()

        progress = 100.0 * ti / t[-1]
        print(f"Progress: {progress:5.1f}%", end="\r")

    plotter_nominal.close()
    plotter_post.close()

    return t, {k: np.asarray(v) for k, v in data.items()}


def main():
    np.random.seed(42)

    params = SimParams()
    
    t, data = run_sim(params, do_baseline=True, plot=True)

    if len(data['p_baseline']) > 0:
        setup_plt()
        plt.figure()
        baseline_err = 1000 * np.linalg.norm(data['p_nominal'] - data['p_baseline'], axis=1)
        plt.plot(t, baseline_err, label=f"baseline (mean={np.mean(baseline_err):.3f} mm)")
        plt.xlabel("time (sec)")
        plt.ylabel("baseline error (mm)")
        plt.grid(True, alpha=0.25)
        plt.legend()

        plt.tight_layout()
        plt.savefig("output/figures/tendon_robot_baseline.pdf", dpi=300)
        plt.close()

    color_cycle = ['r', 'g', 'b', 'c']

    setup_plt(width=3.7, height=3, grid=True)

    fig, axes = plt.subplots(3, 2, sharex=True)
    axes = axes.flatten()  # flatten to 1D for easy indexing

    for ii in range(3):
        ax = axes[ii*2]  # left column
        ax.plot(t, 1000 * data['p_goal'][:, ii], 'k--', label='desired')
        ax.plot(t, 1000 * data['p_meas'][:, ii], linestyle='-', color=color_cycle[ii], label='tracking')
        ax.plot(t, 1000 * data['p_nominal'][:, ii], linestyle=':', color=color_cycle[ii], label='OL')
        ax.set_xlim([t[0], t[-1]+1e-1])
        if ii == 1:
            ax.legend(ncol=3, columnspacing=0.2, borderpad=0.0, borderaxespad=0.2,
                    handlelength=1.0, handletextpad=0.2)

    # plot forces in second column (axes[1], axes[3], axes[5])
    for ii in range(3):
        ax = axes[ii*2+1]  # right column
        ax.plot(t, data['f_gt'][:, ii], 'k--', label='truth')
        ax.plot(t, data['f_mean'][:, ii], color=color_cycle[ii], label='mean')
        ax.fill_between(t, 
            data['f_mean'][:, ii] - 2 * data['f_std'][:, ii],
            data['f_mean'][:, ii] + 2 * data['f_std'][:, ii], 
            alpha=0.2, color=color_cycle[ii], interpolate=True, label=r'2-$\sigma$')
        ax.set_xlim([t[0], t[-1]+1e-1])
        if ii == 2:
            ax.legend(ncol=3, columnspacing=0.2, borderpad=0.0, borderaxespad=0.2,
                    handlelength=1.0, handletextpad=0.2)

    axes[-2].set_xlabel("time (sec)")  # bottom left
    axes[-1].set_xlabel("time (sec)")  # bottom right

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    # Get left and right column centers in figure coordinates
    left_center = (axes[0].get_position().x0 + axes[0].get_position().x1)/2
    right_center = (axes[1].get_position().x0 + axes[1].get_position().x1)/2

    # Place column titles
    fig.text(left_center, 0.97, 'Position (mm)', ha='center', va='bottom', fontsize=8)
    fig.text(right_center, 0.97, 'Force (N)', ha='center', va='bottom', fontsize=8)

    plt.savefig("output/figures/tendon_robot_results.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()