import numpy as np

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


class CosseratRodBaseline:
    def __init__(self, K_inv, rod_length, num_eval_points=20):

        self.K_se_inv = K_inv[3:,3:]
        self.K_bt_inv = K_inv[:3,:3]
        self.rod_length = rod_length

        self.num_eval_points = num_eval_points
        self.base_stress_guess = np.zeros(6)

    def unpack_states(self, y):
        pose = [] 
        stress = []
        for yi in y.transpose():
            T = np.eye(4)
            T[:3,3] = yi[:3]
            T[:3,:3] = yi[3:12].reshape(3, 3)
            
            pose.append(T)
            stress.append(yi[12:])
       
        return {'pose': np.array(pose), 'stress': np.array(stress)}

    def cosserat_rod_deriv(self, s, y):
        p = y[:3]
        R = y[3:12].reshape(3,3)
        n = y[12:15]
        m = y[15:]
            
        v = self.K_se_inv @ R.T @ n + np.array([0, 0, 1])
        u = self.K_bt_inv @ R.T @ m
            
        p_dot = R @ v
        R_dot = R @ hat(u)
        n_dot = np.zeros(3)
        m_dot = -np.cross(p_dot, n)
            
        y_dot = np.hstack((p_dot, R_dot.flatten(), n_dot, m_dot))
        
        return y_dot

    def integrate(self, x):
        p = np.zeros(3)
        R = np.eye(3)
        n = x[:3]
        m = x[3:]
        y0 = np.hstack((p, R.flatten(), n, m))

        integrated = solve_ivp(
            self.cosserat_rod_deriv,
            (0.0, self.rod_length),
            y0,
            method='DOP853',
            rtol=1e-6,
            atol=1e-12,
            t_eval=np.linspace(0, self.rod_length, self.num_eval_points)
        )

        return self.unpack_states(integrated.y)
    
    def shooting_residual(self, x, tip_wrench):
        states = self.integrate(x)

        tip_stress = states['stress'][-1]

        n = tip_stress[:3]
        m = tip_stress[3:]

        e_force = n - tip_wrench[3:]
        e_moment = m - tip_wrench[:3]

        res = np.hstack([
            e_force, 
            e_moment
        ])

        # print(np.linalg.norm(res))
        return res
    
    def solve(self, tip_wrench=np.zeros(6)):
        result = root(self.shooting_residual, self.base_stress_guess, args=(tip_wrench,), tol=1e-10)
        self.base_stress_guess = result.x
        
        solution = self.integrate(result.x)
        
        return solution
    

    

    
    