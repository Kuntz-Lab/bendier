from collections import deque

import numpy as np

import bendier
from .config import get_base_config


def cov_to_std(cov):
    return np.sqrt(np.diag(cov))


class GaussianProcessNoiseModel:
    def __init__(self, dim, frame_rate, total_time, tau=0.3):
        self.dim = dim
        self.dt = 1.0 / frame_rate
        self.num_steps = int(total_time / self.dt)
        self.tau = tau

        t = np.arange(self.num_steps) * self.dt
        ti, tj = np.meshgrid(t, t, indexing='ij')
        
        K = np.exp(-0.5 * (ti - tj)**2 / tau**2)
        K += 1e-8 * np.eye(self.num_steps)

        L = np.linalg.cholesky(K)

        self.samples = L @ np.random.randn(self.num_steps, dim)
        self.i = 0

    def step(self, cov):
        sample = self.samples[self.i]
        self.i += 1
        Lcov = np.linalg.cholesky(cov)
        return Lcov @ sample
    

def generate_trajectory(position_function, sim_time, damping=5e-2, frame_rate=30):
    config = get_base_config()
    solver = bendier.TendonRobotSolver(config)

    num_steps = int(sim_time * frame_rate)
    tensions_min = 0.1 * np.ones(4)
    tensions_mean = tensions_min.copy()

    position_trajectory = []
    tensions_trajectory = []
    t = []

    tensions_cov = (1e-2) ** 2 * np.eye(4)
    tip_wrench_cov = (1e-3) ** 2 * np.eye(6)
    tip_wrench_mean = np.zeros(6)

    for i in range(num_steps):
        t_i = i / float(frame_rate)

        tensions = bendier.Vector4Gaussian(tensions_mean, tensions_cov)
        tip_wrench = bendier.Vector6Gaussian(tip_wrench_mean, tip_wrench_cov)
        solution = solver.solve(tensions, tip_wrench, None)

        J_position = solution.marginals.J_pose_tensions[3:]
        pose = solution.marginals.rod.states[-1].pose.mean
        p = pose[:3, 3]
        R = pose[:3,:3]

        p_desired = position_function(t_i)
        p_error = R.T @ (p_desired - p)

        JTJ = J_position.T @ J_position
        A = JTJ + (damping**2) * np.eye(JTJ.shape[0])
        b = J_position.T @ p_error
        d_tensions = np.linalg.solve(A, b)

        tensions_mean = np.maximum(tensions_mean + d_tensions, tensions_min)

        position_trajectory.append(p_desired)
        tensions_trajectory.append(tensions_mean)
        t.append(t_i)

    return np.array(t), np.array(position_trajectory), np.array(tensions_trajectory)


def generate_waypoints(num_waypoints, center=(0, 0.175, 0.0), radii=(0.1, 0.05, 0.1)):

    waypoints = []
    for _ in range(num_waypoints):
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        r = np.random.random() ** (1/3)
        point = r * direction * np.array(radii)
        waypoints.append(np.array(center) + point)

    return np.array(waypoints)
    

def generate_waypoint_trajectory(sim_time, frame_rate=30.0, time_per_waypoint=3.0, waypoints=None):
    if waypoints is None:
        num_waypoints = int(sim_time / time_per_waypoint) + 1
        waypoints = generate_waypoints(num_waypoints)

    position_function = lambda t: waypoint_trajectory(t, waypoints)
    t, positions, tensions = generate_trajectory(position_function, sim_time, frame_rate=frame_rate)

    return t, positions, tensions, waypoints


def waypoint_trajectory(t, waypoints, time_per_waypoint=3.0):
    num_segments = len(waypoints) - 1

    segment_index = min(int(t // time_per_waypoint), num_segments)
    next_index = min(segment_index + 1, len(waypoints) - 1)

    alpha = (t % time_per_waypoint) / time_per_waypoint
    return (1 - alpha) * waypoints[segment_index] + alpha * waypoints[next_index]    
        