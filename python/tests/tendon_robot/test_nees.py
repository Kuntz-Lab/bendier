import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

from bendier import TendonRobotSolver, Vector6Gaussian, Vector4Gaussian, Vector3Gaussian

from bendier.plotting.utils import setup_plt
from .config import get_base_config, SimParams
from .utils import generate_waypoint_trajectory, GaussianProcessNoiseModel


def get_nees(mean, cov, true):
    e = mean - true
    return e.T @ np.linalg.solve(cov, e)


def run_sim(params):
    
    config = get_base_config()

    t, _, tensions_traj, _ = generate_waypoint_trajectory(params.sim_time, frame_rate=params.frame_rate)

    simulator = TendonRobotSolver(config)

    # Solver that does the actual inference using tip pose data
    solver = TendonRobotSolver(config)

    # We use correllated noise models here for stable solving, RM correlation later with downsample
    q_noise_model = GaussianProcessNoiseModel(4, params.frame_rate, params.sim_time, tau=params.tau)
    f_noise_model = GaussianProcessNoiseModel(3, params.frame_rate, params.sim_time, tau=params.tau)

    data = {
        'q_mean': [], 'q_cov': [], 'q_gt': [],
        'f_mean': [], 'f_cov': [], 'f_gt': [],
    }

    for i, q_mean in enumerate(tensions_traj):
        f_gt = np.zeros(3) + f_noise_model.step(params.f_prior_cov)
        q_gt = q_mean + q_noise_model.step(params.q_meas_cov)

        solution_sim = simulator.solve(
            Vector4Gaussian(q_gt, params.small_q_cov),
            Vector6Gaussian(np.hstack((np.zeros(3), f_gt)), params.small_wrench_cov),
            None
        )

        # Sample the position
        p_sim = solution_sim.marginals.rod.states[-1].pose.mean[:3,3]
        p_meas = np.random.multivariate_normal(p_sim, params.p_meas_cov)

        # Use the sampled position as a prior on tip pose
        solution_post = solver.solve(
            Vector4Gaussian(q_mean, params.q_meas_cov),
            Vector6Gaussian(np.zeros(6), params.wrench_prior_cov),
            Vector3Gaussian(p_meas, params.p_meas_cov)
        )

        wrench_post = solution_post.marginals.external_wrenches[-1]
        q_post = solution_post.marginals.tensions

        data['q_mean'].append(q_post.mean)
        data['q_cov'].append(q_post.cov)
        data['q_gt'].append(q_gt)
        data['f_mean'].append(wrench_post.mean[3:])
        data['f_cov'].append(wrench_post.cov[3:,3:])
        data['f_gt'].append(f_gt)

        progress = 100.0 * i / len(tensions_traj)
        print(f"Progress: {progress:5.1f}%", end="\r")

    return t, {k: np.asarray(v) for k, v in data.items()}


def plot_nees_hist(ax, nees, dof, xmax=15):
    bins = np.linspace(0, xmax, 25)

    ax.hist(
        nees,
        bins=bins,
        density=True,
        alpha=0.3,
        color='r',
        edgecolor='k',
        linewidth=0.5,
        label='NEES'
    )

    x = np.linspace(0, xmax, 400)
    ax.plot(x, chi2.pdf(x, dof), 'k-', label=r'$\chi^2$ PDF')

    ax.grid(alpha=0.25)
    ax.set_xlim(0, xmax)

    lower = chi2.ppf(0.025, dof)
    upper = chi2.ppf(0.975, dof)
    coverage = np.mean((nees >= lower) & (nees <= upper))

    print(f"95% coverage: {100*coverage:.1f}%")
    print(f"samples: {len(nees)}")
    print("Mean force NEES:", np.mean(nees))


def plot_timeseries(ax, mean, std, true):

    t = np.arange(len(mean))
    ax.plot(t, mean, linewidth=1.0)

    ax.fill_between(
        t,
        mean - 2 * std,
        mean + 2 * std,
        alpha=0.25,
        label=r'2$\sigma$'
    )

    ax.plot(t, true, 'k--', linewidth=1.0, label='gt')
    ax.grid(alpha=0.3)


def main():
    np.random.seed(42)

    params = SimParams()
    params.sim_time = 60.0 * 5
    # params.frame_rate = 10.0
    params.tau = 0.2

    t, data = run_sim(params)

    setup_plt(width=3, height=10.0, grid=True)

    fig, axes = plt.subplots(7, 1, sharex=True)

    for i in range(4):
        plot_timeseries(axes[i],
                        data['q_mean'][:,i],
                        np.sqrt(data['q_cov'][:,i,i]),
                        data['q_gt'][:,i])

        axes[i].set_ylabel(f"tensions {i}")

    for i in range(3):
        plot_timeseries(axes[4 + i],
                        data['f_mean'][:,i],
                        np.sqrt(data['f_cov'][:,i,i]),
                        data['f_gt'][:,i])

        axes[4 + i].set_ylabel(f"force {i}")

    # axes[1].set_xlabel("time")
    #
    # axes[1].set_ylabel("force")

    plt.tight_layout()
    plt.savefig("figures/tendon_robot_nees_timeseries.pdf", bbox_inches="tight")


    q_nees = [get_nees(m, c, t) for m, c, t in zip(data['q_mean'], data['q_cov'], data['q_gt'])]
    f_nees = [get_nees(m, c, t) for m, c, t in zip(data['f_mean'], data['f_cov'], data['f_gt'])]
    
    # Remove temporal coorelation by downsampling
    dt_sim = 1.0 / params.frame_rate
    stride = int(round((2 * params.tau) / dt_sim))
    stride = max(1, stride)

    setup_plt(width=3.5, height=0.8, grid=True)

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    
    plot_nees_hist(axes[0], q_nees[::stride], 4)
    plot_nees_hist(axes[1], f_nees[::stride], 3)

    axes[1].set_xlim([0, 15])
    axes[0].set_xlabel("tendon actuation input")
    axes[1].set_xlabel("external tip force")
    axes[0].set_ylabel("density")
    axes[1].legend()

    plt.savefig("figures/tendon_robot_nees.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()