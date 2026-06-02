import numpy as np
import pyvista as pv

from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.linalg import logm
from scipy.spatial.transform import Rotation


def hat(v):
    return np.array([
        [ 0.0, -v[2],  v[1]],
        [ v[2],  0.0, -v[0]],
        [-v[1],  v[0], 0.0]
    ])


class ParallelRobotSolver:
    def __init__(self, config, num_eval_points=100, plot=False):

        self.K_se_inv = config.K_inv[3:,3:]
        self.K_bt_inv = config.K_inv[:3,:3]
        self.num_eval_points = num_eval_points
        self.plot = plot

        self.start_conditions_guess = np.zeros(6 * 6)

        self.p_init = np.array([p[:3,3] for p in config.base_end_poses])
        self.p_final = np.array([p[:3,3] for p in config.tip_end_poses])

        self.platform_z_offset = self.p_final[0,2]

    def single_rod_deriv(self, s, y):
        R = y[3:12].reshape(3, 3)
        n = y[12:15]
        m = y[15:18]
            
        v = self.K_se_inv @ R.T @ n + np.array([0, 0, 1])
        u = self.K_bt_inv @ R.T @ m
            
        p_dot = R @ v
        R_dot = R @ hat(u)
        n_dot = np.zeros(3)
        m_dot = -np.cross(p_dot, n)
            
        y_dot = np.hstack((p_dot, R_dot.flatten(), n_dot, m_dot))
        
        return y_dot

    def plot_solution(self, rods):
        plotter = pv.Plotter()
        
        p_base = [rod['pose'][0,:3,3] for rod in rods]
        base = pv.Cylinder(center=np.mean(p_base, axis=0), direction=(0, 0, 1), radius=0.3, height=0.01)
        plotter.add_mesh(base, color='silver')

        p_tip = [rod['pose'][-1,:3,3] for rod in rods]
        z_tip = rods[0]['pose'][-1,:3,2]
        tip = pv.Cylinder(center=np.mean(p_tip, axis=0), direction=z_tip, radius=0.15, height=0.01)
        plotter.add_mesh(tip, color='silver', opacity=0.5)

        for rod in rods:
            spline = pv.Spline(rod['pose'][:,:3,3], n_points=200)
            tube = spline.tube(radius=0.005)
            plotter.add_mesh(tube, color='ultramarine', opacity = 0.5)

        plotter.enable_anti_aliasing()
        plotter.add_axes()
        plotter.show()

    def solve(self, rod_lengths, tip_force=np.zeros(3), tip_moment=np.zeros(3)):
        result = root(self.total_shooting_residual, self.start_conditions_guess, args=(rod_lengths, tip_force, tip_moment), tol=1e-10)
        self.start_conditions_guess = result.x
        
        solution = self.integrate_rods(result.x, rod_lengths)

        if self.plot:
            self.plot_solution(solution)
        
        return solution
    
    def unpack_x(self, x):
        def unpack_block(b):
            n = b[:3]
            m = np.array([b[3], b[4], 0])
            theta = b[5]
            R = Rotation.from_rotvec([0, 0, theta]).as_matrix().reshape(9, 1)
            return R, n, m
        
        x = x.reshape(6, 6)
        return [unpack_block(x[i]) for i in range(6)]
    
    def unpack_states(self, y):
        pose = [] 
        stress = []
        for i in range(y.shape[1]):
            T = np.eye(4)
            T[:3,:3] = y[3:12,i].reshape(3, 3)
            T[:3,3] = y[0:3,i]
            pose.append(T)
            stress.append(y[12:18,i])
       
        return {'pose': np.array(pose), 'stress': np.array(stress)}
    
    def integrate_rods(self, x, rod_lengths):
        start_conditions = self.unpack_x(x)
        
        results = []
        for i in range(6):
            y0 = np.hstack([self.p_init[i], start_conditions[i][0].flatten(), start_conditions[i][1], start_conditions[i][2]])
            sol = solve_ivp(
                self.single_rod_deriv,
                (0.0, rod_lengths[i]),
                y0,
                t_eval=np.linspace(0, rod_lengths[i], self.num_eval_points),
                method='DOP853', rtol=1e-10, atol=1e-12
            )
            results.append(self.unpack_states(sol.y))

        return results
    
    def total_shooting_residual(self, x, rod_lengths, tip_force, tip_moment):
        rods = self.integrate_rods(x, rod_lengths)

        pose_ends = np.array([rod['pose'][-1] for rod in rods])  # num_rods, 4, 4
        stress_ends = np.array([rod['stress'][-1] for rod in rods])  # num_rods, 6

        p_ends = pose_ends[:,:3,3]
        R_ends = pose_ends[:,:3,:3]
        n_ends = stress_ends[:,:3]
        m_ends = stress_ends[:,3:]

        e_force = np.sum(n_ends, axis=0) - tip_force
        
        z_offset = R_ends[:,:3,2] * self.platform_z_offset
        sum_point = np.mean(p_ends - z_offset, axis=0)
        moments = np.sum([hat(pi - sum_point) @ ni for pi, ni in zip(p_ends, n_ends)], axis=0)
        e_moment = moments + m_ends.sum(axis=0) - tip_moment

        p0 = p_ends[0]
        R0 = R_ends[0]
        
        e_pos = []
        for i in range(1, 6):
            e_pos.append(
                (p0 - R0 @ self.p_final[0]) -
                (p_ends[i] - R_ends[i] @ self.p_final[i])
            )
       

        e_rot = []
        for i in range(1, 6):
            dR = Rotation.from_matrix(R0).inv() * Rotation.from_matrix(R_ends[i])
            e_rot.append(dR.as_rotvec())

        res = np.hstack([
            e_force, 
            e_moment,
            100.0 * np.hstack(e_pos),
            np.hstack(e_rot)
        ])

        # print(np.linalg.norm(res))
        return res
    