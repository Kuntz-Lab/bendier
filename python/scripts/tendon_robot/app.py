import threading

import numpy as np
import viser
import viser.uplot as uplot

import bendier
from bendier.visualization import TendonRobotPlotter
from bendier.visualization.tendon_robot_plotter import TENDON_COLORS as _PLOTTER_TENDON_COLORS

from config import get_config, get_dexterous_tendon_input


def get_app_config(**overrides):
    if "base" not in overrides:
        base = bendier.SolverBaseConfig()
        base.linear_solver_type = "MULTIFRONTAL_CHOLESKY"
        overrides["base"] = base
        overrides["tendon_stiffness"] = 200.0e9 * (np.pi  / 4.0) * (0.0005)**2  # Stainless wire
    return get_config(tendon_input=get_dexterous_tendon_input(), **overrides)


TENDON_COLORS = tuple("#%02x%02x%02x" % c for c in _PLOTTER_TENDON_COLORS)
_GAUSSIAN_PLOT_SCALES = {"x": uplot.Scale(time=False)}
_GAUSSIAN_PLOT_AXES = (uplot.Axis(), uplot.Axis(show=False))
_GAUSSIAN_PLOT_LEGEND = uplot.Legend(show=False)

FORCE_MIN, FORCE_MAX, FORCE_STEP = -0.5, 0.5, 0.01
SIGMA_MOMENT_PRIOR = 0.001
TENSIONS_PRIOR_MEAN = 2.0
TENSIONS_PRIOR_SIGMA = 1.0

DISPLACEMENT_SIGMA_MIN, DISPLACEMENT_SIGMA_MAX, DISPLACEMENT_SIGMA_STEP = 0.0001, 0.01, 0.0001
DISPLACEMENT_SIGMA_INITIAL = 0.0005
FORCE_SIGMA_MIN, FORCE_SIGMA_MAX, FORCE_SIGMA_STEP = 0.0001, 0.2, 0.0001
FORCE_SIGMA_INITIAL = 0.001
IK_DAMPING = 0.5
NULL_SPACE_MIN, NULL_SPACE_MAX, NULL_SPACE_STEP = -0.02, 0.02, 0.0005


def damped_gauss_newton_step(J, error, damping):
    JTJ = J.T @ J
    A = JTJ + (damping ** 2) * np.eye(JTJ.shape[0])
    b = J.T @ error
    return np.linalg.solve(A, b)


# Bounds the position-hold loop (see _solve_holding_position). Each round is
# one more solve, so this is a real cost per tick, but worth spending
# several rounds to actually converge rather than giving up after one.
POSITION_HOLD_MAX_ROUNDS = 8

# Stop condition for the position-hold loop. 0.1 mm -- tight enough to feel
# genuinely "held," loose enough not to chase floating-point noise forever.
POSITION_HOLD_TOLERANCE = 1e-4


def null_space_vector(J, prev_vec=None):
    """Unit vector spanning the null space of J (3x4 here -- 4 tendons is one
    DOF more than the 3D tip position they control, the classic redundant
    "internal tension" freedom of an overactuated cable-driven mechanism,
    now reached via displacement inputs rather than commanded tension).
    Moving displacements along this direction leaves the tip position
    unchanged to first order. SVD doesn't fix a sign for the null vector, so
    this flips it to stay continuous with prev_vec across successive calls
    -- otherwise the *same* button could push displacements in opposite
    physical directions between clicks as the Jacobian evolves.
    """
    _, _, Vt = np.linalg.svd(J, full_matrices=True)
    null_vec = Vt[-1]
    if prev_vec is not None and np.dot(null_vec, prev_vec) < 0:
        null_vec = -null_vec
    return null_vec


def gaussian_curves_for_uplot(means, stds, n_points=150, span_sigmas=4.0):
    """Shared x-axis plus one Gaussian curve per series, for viser's
    add_uplot. Deliberately unnormalized (peak height 1, not a true density
    integrating to 1): these tendons can have wildly different uncertainty
    scales (e.g. a tightly-commanded displacement next to a loosely-inferred
    tension), and a true density would render the tight one as an invisible
    spike while flattening the loose one to near-zero on a shared y-axis --
    peak-height-1 curves keep mean/relative-width comparable at a glance
    regardless of scale. X-axis dynamically spans every series' mean +/-
    span_sigmas*std so all curves stay visible every update, not just
    whichever range happened to be right for one particular tick.
    """
    means = np.asarray(means, dtype=float)
    stds = np.maximum(np.asarray(stds, dtype=float), 1e-9)
    lo = np.min(means - span_sigmas * stds)
    hi = np.max(means + span_sigmas * stds)
    if hi <= lo:
        hi = lo + 1e-6
    x = np.linspace(lo, hi, n_points)
    ys = [np.exp(-0.5 * ((x - m) / s) ** 2) for m, s in zip(means, stds)]
    return x, ys


class TendonRobotApp:
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

        # Target for "Hold Position" mode -- see _solve_holding_position and
        # _ik_step. Set whenever the IK gizmo is dragged (regardless of
        # whether hold mode is currently on, so turning it on later picks up
        # from wherever the tip last was); only *acted* on while the
        # checkbox is checked.
        self._held_position = None

        self.num_tendons = len(get_dexterous_tendon_input().params)

        self.tensions_prior = bendier.VectorXGaussian(
            mean=np.full(self.num_tendons, TENSIONS_PRIOR_MEAN),
            cov=np.diag(np.full(self.num_tendons, TENSIONS_PRIOR_SIGMA ** 2))
        )

        # Displacement has no GUI slider anymore (see the read-only Gaussian
        # plot in _build_gui) -- this is the actual state IK/null-space
        # steps read and write (via _apply_displacement), mirroring what
        # slider.value used to store.
        self._current_displacement = np.zeros(self.num_tendons)

        self._build_gui()
        self.solve_and_update()

    def _build_gui(self):
        server = self.server

        with server.gui.add_folder("Tendon Displacements"):
            self.displacement_plot = server.gui.add_uplot(
                data=tuple([np.zeros(2)] + [np.zeros(2)] * self.num_tendons),
                series=tuple([{}] + [
                    {"label": f"tendon {i}", "stroke": TENDON_COLORS[i], "width": 2}
                    for i in range(self.num_tendons)
                ]),
                scales=_GAUSSIAN_PLOT_SCALES,
                axes=_GAUSSIAN_PLOT_AXES,
                legend=_GAUSSIAN_PLOT_LEGEND,
                title="displacement (m)",
                height=140,
            )
            self.null_space_slider = server.gui.add_slider(
                "null space", min=NULL_SPACE_MIN, max=NULL_SPACE_MAX,
                step=NULL_SPACE_STEP, initial_value=0.0,
                hint="Redundant DOF: drag to shift tendon displacements without moving the tip")
            reset_null_space = server.gui.add_button("Center null space")

        with server.gui.add_folder("Tendon Tensions"):
            # Same treatment as displacement above -- where each tendon
            # sits relative to zero (a real cable can't push), visually,
            # at a glance.
            self.tension_plot = server.gui.add_uplot(
                data=tuple([np.zeros(2)] + [np.zeros(2)] * self.num_tendons),
                series=tuple([{}] + [
                    {"label": f"tendon {i}", "stroke": TENDON_COLORS[i], "width": 2}
                    for i in range(self.num_tendons)
                ]),
                scales=_GAUSSIAN_PLOT_SCALES,
                axes=_GAUSSIAN_PLOT_AXES,
                legend=_GAUSSIAN_PLOT_LEGEND,
                title="tension (N)",
                height=140,
            )

        with server.gui.add_folder("Tip Force"):
            self.force_sliders = [
                server.gui.add_slider(
                    label, min=FORCE_MIN, max=FORCE_MAX,
                    step=FORCE_STEP, initial_value=0.0)
                for label in ("fx", "fy", "fz")
            ]
            reset_wrench = server.gui.add_button("Reset force")

        with server.gui.add_folder("Position Hold"):
            self.hold_position_checkbox = server.gui.add_checkbox(
                "hold position", initial_value=False,
                hint="Keep the controller correcting displacement to hold the tip in place, "
                     "even as tip force changes -- otherwise force is free to move the tip.")

        with server.gui.add_folder("Uncertainty (sigma)"):
            self.displacement_sigma_slider = server.gui.add_slider(
                "displacement sigma", min=DISPLACEMENT_SIGMA_MIN, max=DISPLACEMENT_SIGMA_MAX,
                step=DISPLACEMENT_SIGMA_STEP, initial_value=DISPLACEMENT_SIGMA_INITIAL)
            self.force_sigma_slider = server.gui.add_slider(
                "force sigma", min=FORCE_SIGMA_MIN, max=FORCE_SIGMA_MAX,
                step=FORCE_SIGMA_STEP, initial_value=FORCE_SIGMA_INITIAL)

        with server.gui.add_folder("Solution"):
            self.tip_position_readout = server.gui.add_text(
                "tip position", initial_value="", disabled=True)
            self.status_readout = server.gui.add_text(
                "status", initial_value="ok", disabled=True)
            reset_solver = server.gui.add_button("Reset solver")

        sigma_sliders = [self.displacement_sigma_slider, self.force_sigma_slider]
        for slider in self.force_sliders + sigma_sliders:
            slider.on_update(lambda _: self.solve_and_update())

        # Draggable IK target -- rotation disabled since 4 tendons only ever
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

        @self.hold_position_checkbox.on_update
        def _(_):
            if self.hold_position_checkbox.value and self._held_position is None \
                    and self._last_solution is not None:
                # Nothing dragged yet -- lock onto wherever the tip
                # currently is rather than snapping somewhere unexpected.
                self._held_position = \
                    self._last_solution.marginals.rod.states[-1].pose.mean[:3, 3].copy()
            self.solve_and_update()

        self.null_space_slider.on_update(lambda _: self._null_space_step())

        @reset_null_space.on_click
        def _(_):
            # Just re-centers the slider/reference point -- doesn't touch
            # displacements, since "undo the null-space motion applied so
            # far" isn't well-defined (the null direction itself has moved).
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
        self._suppress_slider_solve = True
        try:
            for slider, value in slider_value_pairs:
                slider.value = value
        finally:
            self._suppress_slider_solve = False
        self.solve_and_update()

    def _apply_displacement(self, displacement):
        # Displacement has no GUI slider to write to anymore (see the
        # Tendon Displacements plot) -- this updates the internal state
        # IK/null-space steps mutate, then solves. Mirrors what
        # _set_sliders used to do for the old displacement sliders.
        self._current_displacement = np.asarray(displacement, dtype=float)
        self.solve_and_update()

    def _reset_solver(self):
        # Recovers from a solver stuck in a bad/diverged state (failed solve,
        # or just a pathological slider combination) by throwing the solver
        # away and starting from a fresh one -- cheaper and more reliable
        # than trying to repair whatever internal state it's in.
        with self._solve_lock:
            self.solver = bendier.TendonRobotSolver(get_app_config())
            self.plotter.reset_solve_stats()
            self._null_space_prev_vec = None
            self._null_space_prev_value = 0.0
            self._held_position = None
            self._current_displacement = np.zeros(self.num_tendons)
            self._set_sliders(
                [(s, 0.0) for s in self.force_sliders]
                + [(self.null_space_slider, 0.0)]
                + [(self.displacement_sigma_slider, DISPLACEMENT_SIGMA_INITIAL)]
                + [(self.force_sigma_slider, FORCE_SIGMA_INITIAL)])

    def _solve_holding_position(self, tip_wrench, displacement_cov):
        displacement_meas = bendier.VectorXGaussian(self._current_displacement, displacement_cov)
        solution = self.solver.solve(self.tensions_prior, tip_wrench, None, displacement_meas)

        if self.hold_position_checkbox.value and self._held_position is not None:
            for _ in range(POSITION_HOLD_MAX_ROUNDS):
                pose_mean = solution.marginals.rod.states[-1].pose.mean
                R, p_meas = pose_mean[:3, :3], pose_mean[:3, 3]
                p_error = R.T @ (self._held_position - p_meas)

                if np.linalg.norm(p_error) < POSITION_HOLD_TOLERANCE:
                    break

                J_position = solution.marginals.J_pose_displacements[3:]
                dq = damped_gauss_newton_step(J_position, p_error, IK_DAMPING)
                self._current_displacement = self._current_displacement + dq

                displacement_meas = bendier.VectorXGaussian(self._current_displacement, displacement_cov)
                solution = self.solver.solve(self.tensions_prior, tip_wrench, None, displacement_meas)

        return solution

    def solve_and_update(self):
        with self._solve_lock:
            if self._suppress_slider_solve:
                return

            force_mean = np.array([s.value for s in self.force_sliders])
            tip_wrench_mean = np.concatenate([np.zeros(3), force_mean])

            displacement_cov = (self.displacement_sigma_slider.value ** 2) * np.eye(self.num_tendons)
            wrench_sigma = np.concatenate([
                np.full(3, SIGMA_MOMENT_PRIOR),
                np.full(3, self.force_sigma_slider.value),
            ])
            wrench_cov = np.diag(wrench_sigma ** 2)
            tip_wrench = bendier.Vector6Gaussian(tip_wrench_mean, wrench_cov)

            try:
                solution = self._solve_holding_position(tip_wrench, displacement_cov)
            except Exception as e:
                print(f"[tendon_robot/app] solve() failed, resetting solver: {e}")
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

            if solution.meta.iterations >= 100:
                self.status_readout.value = (
                    "did not converge -- commanded displacement is (near-)ill-posed")
            else:
                self.status_readout.value = "ok"

            disp_mean = solution.marginals.displacements.mean
            disp_std = np.sqrt(np.maximum(np.diag(solution.marginals.displacements.cov), 0.0))
            x, ys = gaussian_curves_for_uplot(disp_mean, disp_std)
            self.displacement_plot.data = tuple([x, *ys])

            tension_mean = solution.marginals.tensions.mean
            tension_std = np.sqrt(np.maximum(np.diag(solution.marginals.tensions.cov), 0.0))
            x2, ys2 = gaussian_curves_for_uplot(tension_mean, tension_std)
            self.tension_plot.data = tuple([x2, *ys2])

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

            self._held_position = np.asarray(p_goal, dtype=float).copy()

            J_position = self._last_solution.marginals.J_pose_displacements[3:]

            pose_mean = self._last_solution.marginals.rod.states[-1].pose.mean
            R, p_meas = pose_mean[:3, :3], pose_mean[:3, 3]
            p_error = R.T @ (p_goal - p_meas)

            dq = damped_gauss_newton_step(J_position, p_error, IK_DAMPING)
            self._apply_displacement(self._current_displacement + dq)

    def _null_space_step(self):
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

            J_position = self._last_solution.marginals.J_pose_displacements[3:]
            null_vec = null_space_vector(J_position, self._null_space_prev_vec)
            self._null_space_prev_vec = null_vec

            self._apply_displacement(self._current_displacement + delta * null_vec)

            self._ik_step(p_target)


def main():
    server = viser.ViserServer()
    print("Open the URL above in a browser, then drag the sliders in the GUI panel.")
    TendonRobotApp(server)
    server.sleep_forever()


if __name__ == "__main__":
    main()
