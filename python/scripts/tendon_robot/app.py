# TODO: review this file
import threading

import numpy as np
import viser

import bendier
from bendier.visualization import TendonRobotPlotter

# Running this file directly (`python app.py`) puts its own directory first
# on sys.path automatically, so config.py -- right next to this file -- is
# importable with no manual path setup.
from config import get_config, get_dexterous_tendon_input


# Matches SolverBaseConfig's C++ default -- made explicit here (rather than
# left implicit) so solve_and_update() can compare solution.meta.iterations
# against it to detect non-convergence: the optimizer only ever reaches this
# cap by being cut off before satisfying its own stopping criteria, which for
# a well-posed command it reaches in a handful of iterations. Hitting the cap
# means the commanded displacement combination is (near-)ill-posed, not that
# it just needs a bigger iteration budget -- see get_app_config()'s docstring.
SOLVER_MAX_ITERATIONS = 100


def get_app_config(**overrides):
    # The app uses a different (more dexterous) tendon routing than the
    # batch sims/tests -- see get_dexterous_tendon_input()'s docstring.
    # Everything else (rod length, discs, K_inv, sigma_*) stays the same.
    if "base" not in overrides:
        base = bendier.SolverBaseConfig()
        base.max_iterations = SOLVER_MAX_ITERATIONS
        overrides["base"] = base
    return get_config(tendon_input=get_dexterous_tendon_input(), **overrides)

# Displacement (cable payout at the base actuator) is the physically
# realizable command for a real tendon robot -- tension isn't directly
# commandable, it's inferred. Bounds swept empirically across the old
# [0, 10] N tension range for this robot's routing.
DISPLACEMENT_BOUNDS = [
    [-0.03, 0.08],
    [-0.03, 0.08],
    [-0.03, 0.08],
    [-0.03, 0.08],
]

DISPLACEMENT_STEP = 0.0005

FORCE_MIN, FORCE_MAX, FORCE_STEP = -0.5, 0.5, 0.01

# Moments aren't exposed as sliders -- pinned to zero with a small fixed
# sigma so they stay negligible without cluttering the GUI.
MOMENT_SIGMA_FIXED = 0.001

# Sigma sliders: how tightly each prior pins the solve to its mean. Small
# sigma is near-deterministic (the classic forward-mechanics case); larger
# sigma lets the estimate drift away from the slider value in response to
# other factors in the graph -- which is the actual Bayesian behavior on
# display, not just a cosmetic knob.
DISPLACEMENT_SIGMA_MIN, DISPLACEMENT_SIGMA_MAX, DISPLACEMENT_SIGMA_STEP = 0.0001, 0.01, 0.0001
DISPLACEMENT_SIGMA_INITIAL = 0.0005
FORCE_SIGMA_MIN, FORCE_SIGMA_MAX, FORCE_SIGMA_STEP = 0.0001, 0.2, 0.0001
FORCE_SIGMA_INITIAL = 0.001

# Tension is no longer commanded -- it floats free, inferred from the
# displacement-constraint physics and wrench balance. This fixed, broad
# sigma keeps that prior uninformative without leaving it fully flat (which
# can make the linear solve ill-conditioned).
TENSION_FREE_SIGMA = 20.0

# Damped least-squares step for the IK gizmo. Displacement-space is much
# smaller-scale (and, empirically, closer to singular near the straight
# rest state -- observed singular values [17, 13, 0.003] there) than the
# tension-space Jacobian test_tip_force.py's 1e-2 damping was tuned for, so
# this needs to be substantially larger to keep the near-singular direction
# from producing a huge, bound-blowing step.
IK_DAMPING = 0.5

# Null-space slider is a *relative* control -- see _null_space_step. Its
# absolute position has no fixed physical meaning (the null direction itself
# shifts as displacements change), only drags away from wherever it last was.
# Scaled to displacement's much smaller range (~0.11 m total, vs. the old
# tension range of 10 N) -- the old (-3, 3) bounds were tuned for tension
# and would blow through the entire displacement range in a single drag.
NULL_SPACE_MIN, NULL_SPACE_MAX, NULL_SPACE_STEP = -0.02, 0.02, 0.0005


def damped_gauss_newton_step(J, error, damping):
    JTJ = J.T @ J
    A = JTJ + (damping ** 2) * np.eye(JTJ.shape[0])
    b = J.T @ error
    return np.linalg.solve(A, b)


# Floor (N) enforced by clip_step_for_tension_nonneg -- deliberately a small
# *negative* number, not zero or a positive margin. The resting, unloaded
# state has every tension already at ~0, so comparing against a floor at or
# above zero would treat literally the first step away from rest as a full
# violation and freeze every slider permanently at its starting value.
# Kept close to zero -- a looser floor tolerates each individual incremental
# drag step (e.g. a continuous null-space or slider drag fires many small
# steps), but those per-step violations compound: confirmed empirically that
# -0.2 let a 10-step drag walk tension down to -0.16 cumulatively, well past
# what "close to zero" should mean, since each individual step looked small
# enough to pass on its own.
TENSION_SAFETY_FLOOR = -0.02


def clip_step_for_tension_nonneg(delta, tension_current, J_tension_displacement, floor=TENSION_SAFETY_FLOOR):
    """Scale a proposed displacement delta down (uniformly, preserving
    direction) so that, per the current linearized J_tension_displacement
    sensitivity (d tension / d displacement), no tension is predicted to
    drop below `floor`. This is the same "fraction to the boundary" idea an
    interior-point method uses to limit a step -- an approximation (the true
    relationship is nonlinear, and empirically quite strongly so in some
    regions of this system), not a hard guarantee. It's the only line of
    defense against negative tension: the solver itself has no constraint on
    tension sign. Returns delta unchanged if no violation is predicted.
    """
    predicted_delta = J_tension_displacement @ delta
    alpha = 1.0
    for t, dt in zip(tension_current, predicted_delta):
        if dt < 0:
            # Largest alpha in [0, 1] with t + alpha*dt >= floor.
            alpha_i = (floor - t) / dt
            alpha = min(alpha, alpha_i)
    return max(alpha, 0.0) * delta


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


def null_space_tension_correction(null_vec, J_tension, predicted_tension, floor=TENSION_SAFETY_FLOOR):
    """Classical null-space redundancy resolution: given a primary task step
    has already been taken (predicted_tension reflects its effect on
    tension), return the additional alpha*null_vec correction that fixes the
    single worst predicted tension violation, if any. Moving along null_vec
    is exactly the direction that leaves the primary task's output unchanged
    to first order, so this is a "free" secondary objective -- it never
    fights the primary task the way a competing penalty term would.

    Only corrects the worst violation (not a general multi-constraint
    solve): with one redundant DOF, a single scalar generally can't satisfy
    several simultaneous violations that need opposite-signed corrections
    anyway, so there is no real accuracy given up by keeping this simple.
    """
    sensitivity = J_tension @ null_vec
    worst = np.argmin(predicted_tension - floor)
    t, s = predicted_tension[worst], sensitivity[worst]
    if t >= floor or s == 0:
        return np.zeros_like(null_vec)
    alpha = (floor - t) / s
    return alpha * null_vec


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

        self.num_tendons = len(get_dexterous_tendon_input().params)

        self._build_gui()
        self.solve_and_update()

    def _build_gui(self):
        server = self.server

        with server.gui.add_folder("Tendon Displacements"):
            self.displacement_sliders = [
                server.gui.add_slider(
                    f"tendon {i}", min=DISPLACEMENT_BOUNDS[i][0], max=DISPLACEMENT_BOUNDS[i][1],
                    step=DISPLACEMENT_STEP, initial_value=0.0)
                for i in range(self.num_tendons)
            ]
            reset_displacements = server.gui.add_button("Reset displacements")
            # 4 tendons controlling a 3D tip position is overactuated by one
            # DOF -- dragging this shifts displacements along that redundant
            # direction (see null_space_vector) without moving the tip, e.g.
            # to trade off internal tension/stiffness. It's a *relative*
            # control (see _null_space_step): drag it any direction, any
            # amount, as many times as you like -- its absolute position
            # doesn't mean anything on its own, only motion away from
            # wherever it last was.
            self.null_space_slider = server.gui.add_slider(
                "null space", min=NULL_SPACE_MIN, max=NULL_SPACE_MAX,
                step=NULL_SPACE_STEP, initial_value=0.0,
                hint="Redundant DOF: drag to shift tendon displacements without moving the tip")
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
            self.displacement_sigma_slider = server.gui.add_slider(
                "displacement sigma", min=DISPLACEMENT_SIGMA_MIN, max=DISPLACEMENT_SIGMA_MAX,
                step=DISPLACEMENT_SIGMA_STEP, initial_value=DISPLACEMENT_SIGMA_INITIAL)
            self.force_sigma_slider = server.gui.add_slider(
                "force sigma", min=FORCE_SIGMA_MIN, max=FORCE_SIGMA_MAX,
                step=FORCE_SIGMA_STEP, initial_value=FORCE_SIGMA_INITIAL)

        with server.gui.add_folder("Solution"):
            self.tip_position_readout = server.gui.add_text(
                "tip position", initial_value="", disabled=True)
            # Tension is inferred, not commanded, when driving from
            # displacement -- a real cable can't push, but nothing in this
            # Gaussian graph enforces tension >= 0, so a commanded
            # displacement combination that isn't actually achievable can
            # produce a negative inferred tension. Surface it instead of
            # silently accepting an unphysical solution.
            self.tension_readout = server.gui.add_text(
                "tensions (inferred)", initial_value="", disabled=True)
            self.status_readout = server.gui.add_text(
                "status", initial_value="ok", disabled=True)
            reset_solver = server.gui.add_button("Reset solver")

        sigma_sliders = [self.displacement_sigma_slider, self.force_sigma_slider]
        for slider in self.displacement_sliders + self.force_sliders + sigma_sliders:
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

        @reset_displacements.on_click
        def _(_):
            self._set_sliders((s, 0.0) for s in self.displacement_sliders)

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
            self.plotter.reset_solve_stats()
            self._null_space_prev_vec = None
            self._null_space_prev_value = 0.0
            self._set_sliders(
                [(s, 0.0) for s in self.displacement_sliders + self.force_sliders]
                + [(self.null_space_slider, 0.0)]
                + [(self.displacement_sigma_slider, DISPLACEMENT_SIGMA_INITIAL)]
                + [(self.force_sigma_slider, FORCE_SIGMA_INITIAL)])

    def solve_and_update(self):
        with self._solve_lock:
            if self._suppress_slider_solve:
                return

            raw_displacement = np.array([s.value for s in self.displacement_sliders])

            # Every path that changes displacement (IK step, null-space step,
            # or a raw slider drag) funnels through here, so clipping here
            # protects all three uniformly: predict the effect of this
            # change on tension using the current linearized sensitivity,
            # and scale it back if it would cross into negative tension --
            # this is the only guard against it, the solver itself places no
            # constraint on tension sign.
            if self._last_solution is not None:
                current_displacement = self._last_solution.marginals.displacements.mean
                current_tension = self._last_solution.marginals.tensions.mean
                J_tension_displacement = self._last_solution.marginals.J_tension_displacements

                delta = raw_displacement - current_displacement
                clipped_delta = clip_step_for_tension_nonneg(
                    delta, current_tension, J_tension_displacement)
                displacement_mean = current_displacement + clipped_delta

                if not np.allclose(displacement_mean, raw_displacement, atol=1e-9):
                    # Snap the sliders back to what's actually being
                    # commanded, so the display never shows a value that
                    # was silently overridden.
                    self._suppress_slider_solve = True
                    try:
                        for slider, value in zip(self.displacement_sliders, displacement_mean):
                            slider.value = value
                    finally:
                        self._suppress_slider_solve = False
            else:
                displacement_mean = raw_displacement

            force_mean = np.array([s.value for s in self.force_sliders])
            tip_wrench_mean = np.concatenate([np.zeros(3), force_mean])

            displacement_cov = (self.displacement_sigma_slider.value ** 2) * np.eye(self.num_tendons)
            wrench_sigma = np.concatenate([
                np.full(3, MOMENT_SIGMA_FIXED),
                np.full(3, self.force_sigma_slider.value),
            ])
            wrench_cov = np.diag(wrench_sigma ** 2)

            # Tension is no longer commanded -- broad/uninformative prior
            # lets it float, inferred from the displacement + wrench-balance
            # physics (see TENSION_FREE_SIGMA).
            tensions = bendier.VectorXGaussian(
                np.zeros(self.num_tendons), (TENSION_FREE_SIGMA ** 2) * np.eye(self.num_tendons))
            tip_wrench = bendier.Vector6Gaussian(tip_wrench_mean, wrench_cov)
            displacement_meas = bendier.VectorXGaussian(displacement_mean, displacement_cov)

            try:
                solution = self.solver.solve(tensions, tip_wrench, None, displacement_meas)
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

            # The optimizer only reaches SOLVER_MAX_ITERATIONS by being cut
            # off before satisfying its own stopping criteria -- a well-posed
            # command converges in far fewer. Hitting the cap means the
            # result below is a stale, not-actually-converged snapshot, not
            # a real answer -- flag it instead of presenting it as "ok".
            if solution.meta.iterations >= SOLVER_MAX_ITERATIONS:
                self.status_readout.value = (
                    "did not converge -- commanded displacement is (near-)ill-posed")
            else:
                self.status_readout.value = "ok"

            # Nothing in this Gaussian graph enforces tension >= 0 -- a real
            # cable can't push, so flag it if a commanded displacement
            # combination implies an unphysical (negative) tension instead
            # of silently accepting the least-squares result. Small negative
            # tolerance absorbs solver/floating-point noise near true zero
            # (e.g. the straight, unloaded rest state) without false-flagging.
            tensions_inferred = solution.marginals.tensions.mean
            tension_str = ", ".join(f"{t:.3f}" for t in tensions_inferred)
            if np.any(tensions_inferred < -1e-3):
                self.tension_readout.value = f"[{tension_str}] N -- NEGATIVE (unphysical)"
            else:
                self.tension_readout.value = f"[{tension_str}] N"

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

            J_position = self._last_solution.marginals.J_pose_displacements[3:]
            J_tension = self._last_solution.marginals.J_tension_displacements
            tension_current = self._last_solution.marginals.tensions.mean

            pose_mean = self._last_solution.marginals.rod.states[-1].pose.mean
            R, p_meas = pose_mean[:3, :3], pose_mean[:3, 3]
            p_error = R.T @ (p_goal - p_meas)

            # Primary task: track the tip position target.
            dq_primary = damped_gauss_newton_step(J_position, p_error, IK_DAMPING)

            # Secondary, "free" correction along the task's own null-space
            # direction (doesn't disturb tip tracking to first order) to
            # pull tension back from the floor if the primary step alone
            # would cross it. Not tied to self._null_space_prev_vec -- this
            # is a fresh one-shot correction each tick, not an accumulating
            # drag, so the sign-continuity concern that matters for the
            # null-space slider doesn't apply here.
            null_vec = null_space_vector(J_position)
            predicted_tension = tension_current + J_tension @ dq_primary
            dq_correction = null_space_tension_correction(null_vec, J_tension, predicted_tension)

            dq = dq_primary + dq_correction
            displacements = np.clip(
                np.array([s.value for s in self.displacement_sliders]) + dq,
                [bound[0] for bound in DISPLACEMENT_BOUNDS], [bound[1] for bound in DISPLACEMENT_BOUNDS])

            self._set_sliders(zip(self.displacement_sliders, displacements))

    def _null_space_step(self):
        # Relative control: the slider's absolute value has no fixed
        # physical meaning (the null direction shifts as displacements
        # change), so each event applies only *how far it moved since last
        # time*, not its raw position -- lets you drag it back and forth
        # freely, any number of times, without needing to return to some
        # "zero".
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

            displacements = np.clip(
                np.array([s.value for s in self.displacement_sliders]) + delta * null_vec,
                [bound[0] for bound in DISPLACEMENT_BOUNDS], [bound[1] for bound in DISPLACEMENT_BOUNDS])

            self._set_sliders(zip(self.displacement_sliders, displacements))

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
    TendonRobotApp(server)
    server.sleep_forever()


if __name__ == "__main__":
    main()
