import numpy as np
from scipy.optimize import root
from scipy.integrate import solve_ivp


def hat(v):
    vx, vy, vz = v
    return np.array([[0, -vz,  vy],
                     [vz,  0, -vx],
                     [-vy, vx,  0]])


def orthogonalize(R):
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


def pack_state(x):
    return np.hstack([
        x['p'], 
        x['R'].reshape(-1), 
        x['n'], 
        x['m']]
    )


def unpack_state(y):
    return {
        'p': y[0:3], 
        'R': y[3:12].reshape(3,3), 
        'n': y[12:15], 
        'm': y[15:18]
    }


class TendonRobotSolver:
    def __init__(self, config, holes):
        self.K_se_inv = np.array(config.K_inv[3:,3:])
        self.K_bt_inv = np.array(config.K_inv[:3,:3])
        self.holes = np.array(holes)

        self.num_discs = config.num_discs
        self.s_discs = np.linspace(0, config.rod_length, config.num_discs)

        # Solving for the external force at each disc, including tip force
        self.x0 = np.zeros(6 * config.num_discs)

    def solve(self, tensions, tip_force):
        # Shooting method to get base states and disc forces/moment
        sol = root(self.compute_residual, self.x0, args=(tensions, tip_force), tol=1e-12)

        if not sol.success:
            raise RuntimeError(f"Root finding failed: {sol.message}")

        # Given solution, integrate robot to get final backbone solution
        states = self.integrate_robot(sol.x)

        # Warm start next solve
        self.x0 = sol.x

        return states
    
    def segment_dynamics(self, s, y):
        state = unpack_state(y)
        R = state['R']
        m = state['m']
        n = state['n']
            
        v = self.K_se_inv @ R.T @ n + np.array([0, 0, 1])
        u = self.K_bt_inv @ R.T @ m

        state_dot = {}
        state_dot['p'] = R @ v
        state_dot['R'] = R @ hat(u)
        state_dot['n'] = np.zeros(3)
        state_dot['m'] = -np.cross(state_dot['p'], n)

        return pack_state(state_dot)

    def integrate_robot(self, x):
        # Unpack force/moment at each disc
        fl = x.reshape(self.num_discs, 6)
        f = fl[:, :3]
        l = fl[:, 3:]

        # Initial pose from gtsam code
        p0 = np.zeros(3)
        R0 = np.array([[-1, 0, 0],
                       [ 0, 0, 1],
                       [ 0, 1, 0]])
        
        # Initialize states for integration, initial stress is the base force
        state0 = {'p': p0, 'R': R0, 'n': -f[0], 'm': -l[0]}
        y0 = pack_state(state0)
        disc_states = [state0]

        for k in range(len(self.s_discs) - 1):
            # Solve and get the solution only at endpoint b, the next disc
            a, b = self.s_discs[k], self.s_discs[k + 1]
            sol = solve_ivp(self.segment_dynamics, [a, b], y0, method='DOP853', t_eval=[b], rtol=1e-10, atol=1e-12, dense_output=True)

            if not sol.success:
                raise RuntimeError(f"Integration failed: {sol.message}")
            
            y_end = sol.y[:,-1]
            state_end = unpack_state(y_end)

            # Apply the disc wrench to this disc for next integration segment
            state_end['n'] -= f[k + 1]
            state_end['m'] -= l[k + 1]
            state_end['R'] = orthogonalize(state_end['R'])

            # Next segment gets the y for the end of this segment
            y0 = pack_state(state_end)
            disc_states.append(state_end)

        return disc_states
    
    def compute_residual(self, x, tensions, tip_force):
        # Given the force at each node x, determine poses at each node
        disc_states = self.integrate_robot(x)  # (num_discs - 1, ...)

        # Compute the tendon wrenches for each disc given the poses (not including base disc)
        f_pred, l_pred = self.compute_disc_wrenches(disc_states, tensions, tip_force) # (num_discs - 1, 3)

        fl = x.reshape(self.num_discs, 6)
        f  = fl[:, :3]
        l  = fl[:, 3:]

        state_end = disc_states[-1]
        
        e_f = (f[1:] - f_pred).reshape(-1)
        e_l = (l[1:] - l_pred).reshape(-1)

        return np.hstack([state_end['n'], state_end['m'], e_f, e_l])
    
    def get_hole_spatial(self, disc_states, disc_idx, tendon_idx):
        return disc_states[disc_idx]['R'] @ self.holes[disc_idx][tendon_idx] + disc_states[disc_idx]['p']
    
    def get_disc_wrench(self, disc_states, tensions, idx_0, idx_1, tendon_idx):
        hole_0 = self.get_hole_spatial(disc_states, idx_0, tendon_idx)
        hole_1 = self.get_hole_spatial(disc_states, idx_1, tendon_idx)
        hole_diff = hole_1 - hole_0

        f = tensions[tendon_idx] * hole_diff / np.linalg.norm(hole_diff)
        l = np.cross(hole_0 - disc_states[idx_0]['p'], f)
        
        return f, l
    
    def compute_disc_wrenches(self, disc_states, tensions, tip_force):
        f_discs, l_discs = [], []

        for disc_idx in range(1, len(disc_states)):
            f = np.zeros(3)
            l = np.zeros(3)
            
            for tendon_idx in range(len(tensions)):
                f_prev, l_prev = self.get_disc_wrench(disc_states, tensions, disc_idx, disc_idx - 1, tendon_idx)

                f += f_prev
                l += l_prev

                if disc_idx + 1 < len(disc_states):
                    f_next, l_next = self.get_disc_wrench(disc_states, tensions, disc_idx, disc_idx + 1, tendon_idx)

                    f += f_next
                    l += l_next

            f_discs.append(f); l_discs.append(l)

        f_discs[-1] += tip_force

        return np.asarray(f_discs), np.asarray(l_discs)