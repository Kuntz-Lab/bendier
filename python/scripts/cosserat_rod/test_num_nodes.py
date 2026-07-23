import matplotlib.pyplot as plt
import numpy as np

import bendier
from bendier.visualization import setup_plt

from config import get_config, DEFAULTS
from benchmark import CosseratRodBaseline


def get_tip_wrench(t):
    f_xy = 1 * np.array([np.cos(0.3 * t), np.sin(0.3 * t)])
    f_z = 0.5 * np.sin(0.5 * t)

    m_xy = 1 * np.array([np.cos(0.2 * t), np.sin(0.2 * t)])
    m_z = 2 * np.sin(0.4 * t)

    return np.hstack((m_xy, m_z, f_xy, f_z))


def simulate_trajectory(num_nodes=0, num_magnus_terms=0, use_baseline=False, sim_time=180.0, frame_rate=3.0):
    if use_baseline:
        solver = CosseratRodBaseline(DEFAULTS['K_inv'], DEFAULTS['rod_length'])
    else:
        config = get_config(num_magnus_terms=num_magnus_terms, num_nodes=num_nodes)
        solver = bendier.CosseratRodSolver(config)
    
    dt = 1.0 / frame_rate
    num_steps = int(sim_time * frame_rate)

    tip_wrench_cov = np.diag(np.hstack((1e-4 * np.ones(3), 1e-3 * np.ones(3))))**2

    tip_poses = []
    solve_times = []

    for step in range(num_steps + 1):
        t = step * dt
        tip_wrench = get_tip_wrench(t)

        if use_baseline:
            solution = solver.solve(tip_wrench)
            pose = solution['pose'][-1]
        else:
            wrench = bendier.Vector6Gaussian(tip_wrench, tip_wrench_cov)
            solution = solver.solve(wrench, None, None)
            pose = solution.marginals.states[-1].pose.mean
            solve_times.append(solution.meta.total_time_ms)

        tip_poses.append(pose)

        progress = 100.0 * step / num_steps
        print(f"num_nodes: {num_nodes}, num_magnus_terms: {num_magnus_terms}, Progress: {progress:5.1f}%", end="\r")

    tip_poses = np.array(tip_poses)
    p = tip_poses[:,:3,3]
    R = tip_poses[:,:3,:3]

    return {'p': p, 'R': R, 't': np.array(solve_times)}


def run_sims(num_nodes, num_magnus_terms):
    baseline = simulate_trajectory(use_baseline=True)     

    mean_errors = np.zeros((len(num_magnus_terms), len(num_nodes)))
    mean_solve_times = np.zeros((len(num_magnus_terms), len(num_nodes)))

    for i in range(len(num_magnus_terms)):
        for j in range(len(num_nodes)):
            this = simulate_trajectory(num_nodes=num_nodes[j], num_magnus_terms=num_magnus_terms[i], use_baseline=False)
            e_mean = np.linalg.norm(this['p'] - baseline['p'], axis=1).mean()
            mean_errors[i, j] = e_mean
            mean_solve_times[i, j] = this['t'].mean()

    return np.array(mean_errors), np.array(mean_solve_times)


def main():
    np.random.seed(42)
    
    # Test num nodes from 5 to 50 by 1 inclusive 
    num_nodes = np.arange(5, 51, 1)
    num_magnus_terms = [1, 4]
    mean_errors, mean_solve_times = run_sims(num_nodes, num_magnus_terms)
    
    # Convert to percent rod length
    L = DEFAULTS["rod_length"]
    constant_percent = 100 * mean_errors[0, :] / L
    linear_percent = 100 * mean_errors[1, :] / L

    setup_plt(width=2.0,height=2.0)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.semilogy(num_nodes, constant_percent, ':k')
    ax1.semilogy(num_nodes, linear_percent, '-k')
    ax1.set_ylabel("tip error (%)")
    ax1.grid(True, alpha=0.3)


    def fmt_exp(x):
        exp = int(np.log10(x))
        return rf'$10^{{{exp:+d}}}$'  # always show sign

    ax1.set_yticks([1e-2, 1e-1, 1, 10])
    ax1.set_yticklabels([fmt_exp(t) for t in [1e-2, 1e-1, 1, 10]])


    ax2.plot(num_nodes, mean_solve_times[0, :], ':k', label="PW constant")
    ax2.plot(num_nodes, mean_solve_times[1, :], '-k', label="PW linear")
    # ax2.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.set_ylabel("solve time (ms)")
    ax2.set_xlabel("number of arclength nodes")
    ax2.set_xticks([10, 20, 30, 40, 50])
    ax2.set_yticks([0, 5, 10])
    ax2.legend(borderpad=0.0, borderaxespad=0.2, handlelength=1.0, handletextpad=0.2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.align_ylabels()

    plt.savefig("output/figures/cosserat_num_nodes.pdf")
    plt.close()


if __name__ == "__main__":
    main()