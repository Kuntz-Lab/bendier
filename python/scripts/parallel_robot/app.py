import threading

import numpy as np
import viser
from scipy.spatial.transform import Rotation

import bendier
from bendier.visualization import ParallelRobotPlotter
from bendier.visualization.utils import pose_to_wxyz

# Running this file directly (`python app.py`) puts its own directory first
# on sys.path automatically, so config.py -- right next to this file -- is
# importable with no manual path setup.
from config import get_config, get_base_poses, platform_z_offset

ROD_LENGTH_MIN, ROD_LENGTH_MAX, ROD_LENGTH_STEP = 0.3, 0.9, 0.005
ROD_LENGTH_INITIAL = 0.6

FORCE_MIN, FORCE_MAX, FORCE_STEP = -3.0, 3.0, 0.05

# Moments aren't exposed as sliders -- pinned to zero with a small fixed
# sigma so they stay negligible without cluttering the GUI.
MOMENT_SIGMA_FIXED = 0.001

# Sigma sliders: how tightly each prior pins the solve to its mean. Small
# sigma is near-deterministic (the classic forward-mechanics case); larger
# sigma lets the estimate drift away from the slider value in response to
# other factors in the graph.
ROD_LENGTHS_SIGMA_MIN, ROD_LENGTHS_SIGMA_MAX, ROD_LENGTHS_SIGMA_STEP = 0.0001, 0.05, 0.0001
ROD_LENGTHS_SIGMA_INITIAL = 0.001
FORCE_SIGMA_MIN, FORCE_SIGMA_MAX, FORCE_SIGMA_STEP = 0.0001, 1.0, 0.0001
FORCE_SIGMA_INITIAL = 0.001

# Damped least-squares step for the IK gizmo. No existing precedent uses
# damping for this robot (python/scripts/parallel_robot/test_tip_force.py
# uses a bare pinv), but an interactively-dragged target can be pushed
# toward a kinematic singularity in a way the open-loop test trajectory
# never does -- reusing the tendon script's damping value as a starting
# point; tune live if a drag near the workspace boundary looks unstable.
IK_DAMPING = 1e-2


def damped_gauss_newton_step(J, error, damping):
    JTJ = J.T @ J
    A = JTJ + (damping ** 2) * np.eye(JTJ.shape[0])
    b = J.T @ error
    return np.linalg.solve(A, b)


class ParallelRobotApp:
    def __init__(self, server: viser.ViserServer):
        self.server = server
        self.solver = bendier.ParallelRobotSolver(get_config())
        self.plotter = ParallelRobotPlotter(
            server, platform_z_offset=platform_z_offset,
            plot_rod_wrenches=False, plot_tip_force=True)
        # viser dispatches client-driven GUI callbacks on a thread pool, so
        # rapidly dragging a slider can fire solve_and_update() from
        # multiple threads at once. The solver isn't safe to call
        # concurrently on the same instance (confirmed: concurrent solve()
        # calls corrupt its internal state and crash), so serialize with a
        # lock. It has to be reentrant: setting a slider's .value from
        # Python (as _reset_solver does) calls on_update synchronously on
        # the *same* thread, which would otherwise deadlock against a plain
        # Lock we're already holding.
        self._solve_lock = threading.RLock()

        # IK gizmo state -- see _ik_step/_reposition_ik_gizmo. _last_solution
        # is the Jacobian source for the next IK tick (always exactly
        # consistent with the current slider values, since nothing mutates
        # them between one solve_and_update() and the next except _ik_step
        # itself, which always re-solves immediately under the same lock).
        self._last_solution = None
        self._gizmo_dragging = False
        self._suppress_slider_solve = False
        self._ik_gizmo = None

        self.num_rods = len(get_base_poses())

        self._build_gui()
        self.solve_and_update()

    def _build_gui(self):
        server = self.server

        with server.gui.add_folder("Rod Lengths"):
            self.rod_length_sliders = [
                server.gui.add_slider(
                    f"rod {i}", min=ROD_LENGTH_MIN, max=ROD_LENGTH_MAX,
                    step=ROD_LENGTH_STEP, initial_value=ROD_LENGTH_INITIAL)
                for i in range(self.num_rods)
            ]
            reset_lengths = server.gui.add_button("Reset lengths")

        with server.gui.add_folder("Platform Force"):
            self.force_sliders = [
                server.gui.add_slider(
                    label, min=FORCE_MIN, max=FORCE_MAX,
                    step=FORCE_STEP, initial_value=0.0)
                for label in ("fx", "fy", "fz")
            ]
            reset_wrench = server.gui.add_button("Reset force")

        with server.gui.add_folder("Uncertainty (sigma)"):
            self.rod_lengths_sigma_slider = server.gui.add_slider(
                "rod lengths sigma", min=ROD_LENGTHS_SIGMA_MIN, max=ROD_LENGTHS_SIGMA_MAX,
                step=ROD_LENGTHS_SIGMA_STEP, initial_value=ROD_LENGTHS_SIGMA_INITIAL)
            self.force_sigma_slider = server.gui.add_slider(
                "force sigma", min=FORCE_SIGMA_MIN, max=FORCE_SIGMA_MAX,
                step=FORCE_SIGMA_STEP, initial_value=FORCE_SIGMA_INITIAL)

        with server.gui.add_folder("Solution"):
            self.platform_position_readout = server.gui.add_text(
                "platform position", initial_value="", disabled=True)
            self.status_readout = server.gui.add_text(
                "status", initial_value="ok", disabled=True)
            reset_solver = server.gui.add_button("Reset solver")

        sigma_sliders = [self.rod_lengths_sigma_slider, self.force_sigma_slider]
        for slider in self.rod_length_sliders + self.force_sliders + sigma_sliders:
            slider.on_update(lambda _: self.solve_and_update())

        # Draggable IK target -- fully actuated (6 rods, 6 DOF), so unlike
        # the tendon app rotation stays enabled.
        self._ik_gizmo = server.scene.add_transform_controls("/ik_target", scale=0.1)

        @self._ik_gizmo.on_update
        def _(event):
            if event.phase == "start":
                self._gizmo_dragging = True
            elif event.phase == "update":
                self._ik_step(np.asarray(event.target.position), np.asarray(event.target.wxyz))
            elif event.phase == "end":
                self._gizmo_dragging = False
                self._reposition_ik_gizmo()

        @reset_lengths.on_click
        def _(_):
            self._set_sliders((s, ROD_LENGTH_INITIAL) for s in self.rod_length_sliders)

        @reset_wrench.on_click
        def _(_):
            self._set_sliders((s, 0.0) for s in self.force_sliders)

        @reset_solver.on_click
        def _(_):
            self._reset_solver()

    def _set_sliders(self, slider_value_pairs):
        # Sets several sliders as one atomic update: suppresses the
        # per-slider solve_and_update() cascade that setting .value from
        # Python would otherwise trigger (each change fires its on_update
        # callback synchronously), so only one solve happens against the
        # fully-updated state. Without this, intermediate partially-updated
        # states get solved individually on the *same* solver instance, and
        # the optimizer warm-starts each next solve off whatever the
        # previous (possibly pathological) intermediate one converged to --
        # confirmed this can leave the solver stuck in a bad local minimum
        # even once every slider has reached its correct final value.
        self._suppress_slider_solve = True
        try:
            for slider, value in slider_value_pairs:
                slider.value = value
        finally:
            self._suppress_slider_solve = False
        self.solve_and_update()

    def _reset_solver(self):
        # Recovers from a solver stuck in a bad/diverged state (failed solve,
        # or just a pathological slider combination) by throwing the solver
        # away and starting from a fresh one -- cheaper and more reliable
        # than trying to repair whatever internal state it's in.
        with self._solve_lock:
            self.solver = bendier.ParallelRobotSolver(get_config())
            self.plotter.reset_solve_stats()
            self._set_sliders(
                [(s, ROD_LENGTH_INITIAL) for s in self.rod_length_sliders]
                + [(s, 0.0) for s in self.force_sliders]
                + [(self.rod_lengths_sigma_slider, ROD_LENGTHS_SIGMA_INITIAL)]
                + [(self.force_sigma_slider, FORCE_SIGMA_INITIAL)])

    def solve_and_update(self):
        with self._solve_lock:
            if self._suppress_slider_solve:
                return

            rod_lengths = np.array([s.value for s in self.rod_length_sliders])
            force_mean = np.array([s.value for s in self.force_sliders])
            wrench_mean = np.concatenate([np.zeros(3), force_mean])

            wrench_sigma = np.concatenate([
                np.full(3, MOMENT_SIGMA_FIXED),
                np.full(3, self.force_sigma_slider.value),
            ])
            wrench_cov = np.diag(wrench_sigma ** 2)
            wrench = bendier.Vector6Gaussian(wrench_mean, wrench_cov)

            try:
                solution = self.solver.solve(
                    rod_lengths, self.rod_lengths_sigma_slider.value, wrench, None)
            except Exception as e:
                print(f"[parallel_robot/app] solve() failed, resetting solver: {e}")
                self.solver = bendier.ParallelRobotSolver(get_config())
                self.status_readout.value = f"solve failed ({type(e).__name__}) -- solver reset"
                # IK's Jacobian source must not point at a stale solve while
                # the sliders already hold the (bad) values that caused the
                # failure -- clearing it makes the next IK tick a no-op
                # instead of compounding a step from mismatched state.
                self._last_solution = None
                return

            self.plotter.update(solution)

            platform_position = solution.marginals.platform_pose.mean[:3, 3]
            self.platform_position_readout.value = (
                f"[{platform_position[0]:.4f}, {platform_position[1]:.4f}, {platform_position[2]:.4f}] m")
            self.status_readout.value = "ok"

            self._last_solution = solution
            if self._ik_gizmo is not None and not self._gizmo_dragging:
                self._reposition_ik_gizmo()

    def _reposition_ik_gizmo(self):
        if self._last_solution is None:
            return
        pose_mean = self._last_solution.marginals.platform_pose.mean
        self._ik_gizmo.position = pose_mean[:3, 3]
        self._ik_gizmo.wxyz = pose_to_wxyz(pose_mean)

    def _ik_step(self, p_goal, wxyz_goal):
        with self._solve_lock:
            if self._last_solution is None:
                return

            pose_mean = self._last_solution.marginals.platform_pose.mean
            R, p = pose_mean[:3, :3], pose_mean[:3, 3]
            w, x, y, z = wxyz_goal
            R_goal = Rotation.from_quat([x, y, z, w]).as_matrix()

            p_error = R.T @ (p_goal - p)
            r_error = Rotation.from_matrix(R.T @ R_goal).as_rotvec()
            twist_error = np.hstack((r_error, p_error))

            J = self._last_solution.marginals.rod_lengths_jacobian
            d_rod_lengths = damped_gauss_newton_step(J, twist_error, IK_DAMPING)
            rod_lengths = np.clip(
                np.array([s.value for s in self.rod_length_sliders]) + d_rod_lengths,
                ROD_LENGTH_MIN, ROD_LENGTH_MAX)

            self._set_sliders(zip(self.rod_length_sliders, rod_lengths))


def main():
    server = viser.ViserServer()
    print("Open the URL above in a browser, then drag the sliders in the GUI panel.")
    ParallelRobotApp(server)
    server.sleep_forever()


if __name__ == "__main__":
    main()
