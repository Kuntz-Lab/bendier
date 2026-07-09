import os
import sys
import threading

import numpy as np
import viser

import bendier
from bendier.visualization import TendonRobotPlotter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "tendon_robot"))
from config import get_config, get_dexterous_tendon_input  # noqa: E402


def get_app_config(**overrides):
    # The app uses a different (more dexterous) tendon routing than the
    # batch sims/tests -- see get_dexterous_tendon_input()'s docstring.
    # Everything else (rod length, discs, K_inv, sigma_*) stays the same.
    return get_config(tendon_input=get_dexterous_tendon_input(), **overrides)

TENSION_BOUNDS = [
    [0.0, 10.0],
    [0.0, 10.0],
    [0.0, 10.0],
    [0.0, 10.0],
]
    
TENSION_STEP = 0.05

FORCE_MIN, FORCE_MAX, FORCE_STEP = -0.5, 0.5, 0.01

# Moments aren't exposed as sliders -- pinned to zero with a small fixed
# sigma so they stay negligible without cluttering the GUI.
MOMENT_SIGMA_FIXED = 0.001

# Sigma sliders: how tightly each prior pins the solve to its mean. Small
# sigma is near-deterministic (the classic forward-mechanics case); larger
# sigma lets the estimate drift away from the slider value in response to
# other factors in the graph -- which is the actual Bayesian behavior on
# display, not just a cosmetic knob.
TENSION_SIGMA_MIN, TENSION_SIGMA_MAX, TENSION_SIGMA_STEP = 0.001, 2.0, 0.001
TENSION_SIGMA_INITIAL = 0.01
FORCE_SIGMA_MIN, FORCE_SIGMA_MAX, FORCE_SIGMA_STEP = 0.0001, 0.2, 0.0001
FORCE_SIGMA_INITIAL = 0.001

# Damped least-squares step for the IK gizmo -- matches
# python/scripts/tendon_robot/test_tip_force.py's control-loop damping.
IK_DAMPING = 1e-2

# Null-space slider is a *relative* control -- see _null_space_step. Its
# absolute position has no fixed physical meaning (the null direction itself
# shifts as tensions change), only drags away from wherever it last was.
NULL_SPACE_MIN, NULL_SPACE_MAX, NULL_SPACE_STEP = -3.0, 3.0, 0.02


def format_solve_stats(meta, avg_total_ms):
    return (
        f"iter: {meta.iterations}   err: {meta.error:.2e}\n"
        f"build: {meta.build_time_ms:.2f} ms   opt: {meta.optimize_time_ms:.2f} ms\n"
        f"marg: {meta.marginalize_time_ms:.2f} ms   extr: {meta.extract_time_ms:.2f} ms\n"
        f"total: {meta.total_time_ms:.2f} ms   avg: {avg_total_ms:.2f} ms"
    )


def damped_gauss_newton_step(J, error, damping):
    JTJ = J.T @ J
    A = JTJ + (damping ** 2) * np.eye(JTJ.shape[0])
    b = J.T @ error
    return np.linalg.solve(A, b)


def null_space_vector(J, prev_vec=None):
    """Unit vector spanning the null space of J (3x4 here -- 4 tendons is one
    DOF more than the 3D tip position they control, the classic redundant
    "internal tension" freedom of an overactuated cable-driven mechanism).
    Moving tensions along this direction leaves the tip position unchanged
    to first order. SVD doesn't fix a sign for the null vector, so this
    flips it to stay continuous with prev_vec across successive calls --
    otherwise the *same* button could push tensions in opposite physical
    directions between clicks as the Jacobian evolves.
    """
    _, _, Vt = np.linalg.svd(J, full_matrices=True)
    null_vec = Vt[-1]
    if prev_vec is not None and np.dot(null_vec, prev_vec) < 0:
        null_vec = -null_vec
    return null_vec


class TendonForwardSimApp:
    def __init__(self, server: viser.ViserServer):
        self.server = server
        self.solver = bendier.TendonRobotSolver(get_app_config())
        self.plotter = TendonRobotPlotter(
            server, plot_tip_force=True, plot_backbone_ellipsoids=True)
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
        self._solve_times = []

        # IK gizmo state -- see _ik_step/_reposition_ik_gizmo. _last_solution
        # is the Jacobian source for the next IK tick (always exactly
        # consistent with the current slider values, since nothing mutates
        # them between one solve_and_update() and the next except _ik_step
        # itself, which always re-solves immediately under the same lock).
        self._last_solution = None
        self._gizmo_dragging = False
        self._suppress_slider_solve = False
        self._ik_gizmo = None
        # Sign-continuity anchor for null_space_vector -- see its docstring.
        self._null_space_prev_vec = None
        # Last-seen value of the null-space slider, so on_update can compute
        # how far it moved *this* event (see _null_space_step) rather than
        # treating its absolute position as meaningful.
        self._null_space_prev_value = 0.0

        self.num_tendons = len(get_dexterous_tendon_input().functions)

        self._build_gui()
        self.solve_and_update()

    def _build_gui(self):
        server = self.server

        with server.gui.add_folder("Tendon Tensions"):
            self.tension_sliders = [
                server.gui.add_slider(
                    f"tendon {i}", min=TENSION_BOUNDS[i][0], max=TENSION_BOUNDS[i][1],
                    step=TENSION_STEP, initial_value=0.0)
                for i in range(self.num_tendons)
            ]
            reset_tensions = server.gui.add_button("Reset tensions")
            # 4 tendons controlling a 3D tip position is overactuated by one
            # DOF -- dragging this shifts tensions along that redundant
            # direction (see null_space_vector) without moving the tip, e.g.
            # to trade off internal tension/stiffness. It's a *relative*
            # control (see _null_space_step): drag it any direction, any
            # amount, as many times as you like -- its absolute position
            # doesn't mean anything on its own, only motion away from
            # wherever it last was.
            self.null_space_slider = server.gui.add_slider(
                "null space", min=NULL_SPACE_MIN, max=NULL_SPACE_MAX,
                step=NULL_SPACE_STEP, initial_value=0.0,
                hint="Redundant DOF: drag to shift tendon tension without moving the tip")
            reset_null_space = server.gui.add_button("Center null space")

        with server.gui.add_folder("Tip Force"):
            self.force_sliders = [
                server.gui.add_slider(
                    label, min=FORCE_MIN, max=FORCE_MAX,
                    step=FORCE_STEP, initial_value=0.0)
                for label in ("fx", "fy", "fz")
            ]
            reset_wrench = server.gui.add_button("Reset force")

        with server.gui.add_folder("Uncertainty (sigma)"):
            self.tension_sigma_slider = server.gui.add_slider(
                "tension sigma", min=TENSION_SIGMA_MIN, max=TENSION_SIGMA_MAX,
                step=TENSION_SIGMA_STEP, initial_value=TENSION_SIGMA_INITIAL)
            self.force_sigma_slider = server.gui.add_slider(
                "force sigma", min=FORCE_SIGMA_MIN, max=FORCE_SIGMA_MAX,
                step=FORCE_SIGMA_STEP, initial_value=FORCE_SIGMA_INITIAL)

        with server.gui.add_folder("Solution"):
            self.tip_position_readout = server.gui.add_text(
                "tip position", initial_value="", disabled=True)
            self.solve_stats_readout = server.gui.add_text(
                "solve stats", initial_value="", disabled=True)
            self.status_readout = server.gui.add_text(
                "status", initial_value="ok", disabled=True)
            reset_solver = server.gui.add_button("Reset solver")

        sigma_sliders = [self.tension_sigma_slider, self.force_sigma_slider]
        for slider in self.tension_sliders + self.force_sliders + sigma_sliders:
            slider.on_update(lambda _: self.solve_and_update())

        # Draggable IK target -- rotation disabled since 4 tensions only ever
        # control a 3D position target (matches test_tip_force.py's control
        # loop, which only ever consumes the position rows of the Jacobian).
        self._ik_gizmo = server.scene.add_transform_controls(
            "/ik_target", scale=0.05, disable_rotations=True)

        @self._ik_gizmo.on_update
        def _(event):
            if event.phase == "start":
                self._gizmo_dragging = True
            elif event.phase == "update":
                self._ik_step(np.asarray(event.target.position))
            elif event.phase == "end":
                self._gizmo_dragging = False
                self._reposition_ik_gizmo()

        @reset_tensions.on_click
        def _(_):
            self._set_sliders((s, 0.0) for s in self.tension_sliders)

        self.null_space_slider.on_update(lambda _: self._null_space_step())

        @reset_null_space.on_click
        def _(_):
            # Just re-centers the slider/reference point -- doesn't touch
            # tensions, since "undo the null-space motion applied so far"
            # isn't well-defined (the null direction itself has moved).
            self._suppress_slider_solve = True
            try:
                self.null_space_slider.value = 0.0
            finally:
                self._suppress_slider_solve = False
            self._null_space_prev_value = 0.0

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
            self.solver = bendier.TendonRobotSolver(get_app_config())
            self._solve_times = []
            self._null_space_prev_vec = None
            self._null_space_prev_value = 0.0
            self._set_sliders(
                [(s, 0.0) for s in self.tension_sliders + self.force_sliders]
                + [(self.null_space_slider, 0.0)]
                + [(self.tension_sigma_slider, TENSION_SIGMA_INITIAL)]
                + [(self.force_sigma_slider, FORCE_SIGMA_INITIAL)])

    def solve_and_update(self):
        with self._solve_lock:
            if self._suppress_slider_solve:
                return

            tensions_mean = np.array([s.value for s in self.tension_sliders])
            force_mean = np.array([s.value for s in self.force_sliders])
            tip_wrench_mean = np.concatenate([np.zeros(3), force_mean])

            tensions_cov = (self.tension_sigma_slider.value ** 2) * np.eye(self.num_tendons)
            wrench_sigma = np.concatenate([
                np.full(3, MOMENT_SIGMA_FIXED),
                np.full(3, self.force_sigma_slider.value),
            ])
            wrench_cov = np.diag(wrench_sigma ** 2)

            tensions = bendier.VectorXGaussian(tensions_mean, tensions_cov)
            tip_wrench = bendier.Vector6Gaussian(tip_wrench_mean, wrench_cov)

            try:
                solution = self.solver.solve(tensions, tip_wrench, None)
            except Exception as e:
                print(f"[tendon_forward_sim] solve() failed, resetting solver: {e}")
                self.solver = bendier.TendonRobotSolver(get_app_config())
                self.status_readout.value = f"solve failed ({type(e).__name__}) -- solver reset"
                # IK's Jacobian source must not point at a stale solve while
                # the sliders already hold the (bad) values that caused the
                # failure -- clearing it makes the next IK tick a no-op
                # instead of compounding a step from mismatched state.
                self._last_solution = None
                return

            self.plotter.update(solution)

            tip_position = solution.marginals.rod.states[-1].pose.mean[:3, 3]
            self.tip_position_readout.value = (
                f"[{tip_position[0]:.4f}, {tip_position[1]:.4f}, {tip_position[2]:.4f}] m")
            self._solve_times.append(solution.meta.total_time_ms)
            self.solve_stats_readout.value = format_solve_stats(
                solution.meta, np.mean(self._solve_times))
            self.status_readout.value = "ok"

            self._last_solution = solution
            if self._ik_gizmo is not None and not self._gizmo_dragging:
                self._reposition_ik_gizmo()

    def _reposition_ik_gizmo(self):
        if self._last_solution is None:
            return
        self._ik_gizmo.position = self._last_solution.marginals.rod.states[-1].pose.mean[:3, 3]

    def _ik_step(self, p_goal):
        with self._solve_lock:
            if self._last_solution is None:
                return

            J_position = self._last_solution.marginals.J_pose_tensions[3:]
            pose_mean = self._last_solution.marginals.rod.states[-1].pose.mean
            R, p_meas = pose_mean[:3, :3], pose_mean[:3, 3]
            p_error = R.T @ (p_goal - p_meas)

            dq = damped_gauss_newton_step(J_position, p_error, IK_DAMPING)
            tensions = np.clip(
                np.array([s.value for s in self.tension_sliders]) + dq,
                [bound[0] for bound in TENSION_BOUNDS], [bound[1] for bound in TENSION_BOUNDS])

            self._set_sliders(zip(self.tension_sliders, tensions))

    def _null_space_step(self):
        # Relative control: the slider's absolute value has no fixed
        # physical meaning (the null direction shifts as tensions change),
        # so each event applies only *how far it moved since last time*,
        # not its raw position -- lets you drag it back and forth freely,
        # any number of times, without needing to return to some "zero".
        if self._suppress_slider_solve:
            return

        with self._solve_lock:
            if self._last_solution is None:
                return

            new_value = self.null_space_slider.value
            delta = new_value - self._null_space_prev_value
            self._null_space_prev_value = new_value
            if delta == 0.0:
                return

            p_target = self._last_solution.marginals.rod.states[-1].pose.mean[:3, 3].copy()

            J_position = self._last_solution.marginals.J_pose_tensions[3:]
            null_vec = null_space_vector(J_position, self._null_space_prev_vec)
            self._null_space_prev_vec = null_vec

            tensions = np.clip(
                np.array([s.value for s in self.tension_sliders]) + delta * null_vec,
                [bound[0] for bound in TENSION_BOUNDS], [bound[1] for bound in TENSION_BOUNDS])

            self._set_sliders(zip(self.tension_sliders, tensions))

            # The null-space step only holds the tip exactly at first order --
            # over a long continuous drag the null direction itself rotates
            # and the tip can creep. Cancel that by pulling back toward
            # where the tip was right before this tick, reusing the same
            # IK correction the drag gizmo uses (so it stays locked across
            # the whole drag, not just for infinitesimal moves).
            self._ik_step(p_target)


def main():
    server = viser.ViserServer()
    print("Open the URL above in a browser, then drag the sliders in the GUI panel.")
    TendonForwardSimApp(server)
    server.sleep_forever()


if __name__ == "__main__":
    main()
