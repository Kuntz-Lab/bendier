import os
import sys
import threading

import numpy as np
import viser
from viser.extras import ViserUrdf

import bendier
from bendier.visualization import RigidRobotPlotter
from bendier.visualization import utils as viz_utils

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "rigid_robot"))
from config import (  # noqa: E402
    load_urdf, home_config, build_joint_specs, build_base_calibration, build_tip_offset_calibration,
    DEFAULT_OFFSET_SIGMA_ROT, DEFAULT_OFFSET_SIGMA_POS)

JOINT_STEP = 0.01

JOINT_SIGMA_MIN, JOINT_SIGMA_MAX, JOINT_SIGMA_STEP = 0.0001, 0.3, 0.0001
JOINT_SIGMA_INITIAL = 0.01

OFFSET_SIGMA_ROT_MIN, OFFSET_SIGMA_ROT_MAX, OFFSET_SIGMA_ROT_STEP = 1e-5, 0.05, 1e-5
OFFSET_SIGMA_POS_MIN, OFFSET_SIGMA_POS_MAX, OFFSET_SIGMA_POS_STEP = 1e-5, 0.02, 1e-5

FORCE_MIN, FORCE_MAX, FORCE_STEP = -20.0, 20.0, 0.1
MOMENT_MIN, MOMENT_MAX, MOMENT_STEP = -10.0, 10.0, 0.05

TIP_FORCE_SIGMA_MIN, TIP_FORCE_SIGMA_MAX, TIP_FORCE_SIGMA_STEP = 0.001, 2.0, 0.001
TIP_FORCE_SIGMA_INITIAL = 0.05
TIP_MOMENT_SIGMA_MIN, TIP_MOMENT_SIGMA_MAX, TIP_MOMENT_SIGMA_STEP = 0.001, 2.0, 0.001
TIP_MOMENT_SIGMA_INITIAL = 0.05

TIP_POSE_SIGMA_ROT_MIN, TIP_POSE_SIGMA_ROT_MAX, TIP_POSE_SIGMA_ROT_STEP = 0.0001, 1.0, 0.0001
# Tight by default so dragging the gizmo feels authoritative (close to hard
# IK) out of the box -- loosen these sliders to see the soft MAP-compromise
# behavior against the joint-value prior instead.
TIP_POSE_SIGMA_ROT_INITIAL = 0.001
TIP_POSE_SIGMA_POS_MIN, TIP_POSE_SIGMA_POS_MAX, TIP_POSE_SIGMA_POS_STEP = 0.00001, 0.5, 0.00001
TIP_POSE_SIGMA_POS_INITIAL = 0.0005

# Dedicated, near-hard sigma for holding the tip fixed during null-space
# stepping -- deliberately *not* tied to the tip_pose_sigma_* sliders above
# (those are for how authoritative the gizmo feels, a separate purpose).
# Null-space motion is only a first-order self-motion, so without a strong
# hold correcting the residual each step, drift compounds.
NULL_SPACE_HOLD_SIGMA_ROT = 1e-5
NULL_SPACE_HOLD_SIGMA_POS = 1e-5

# Null-space slider is a *relative* control -- see _null_space_step. Its
# absolute position has no fixed physical meaning (the null direction itself
# shifts as joint values change), only drags away from wherever it last was.
NULL_SPACE_MIN, NULL_SPACE_MAX, NULL_SPACE_STEP = -3.0, 3.0, 0.02


def null_space_vector(J, prev_vec=None):
    """Unit vector spanning the null space of J (6x7 here -- a 7-DOF arm
    controlling a 6-DOF tip pose is redundant by exactly one DOF, the
    classic "elbow self-motion" of a 7-axis manipulator). Moving joint
    values along this direction leaves the tip pose unchanged to first
    order. SVD doesn't fix a sign for the null vector, so this flips it to
    stay continuous with prev_vec across successive calls -- otherwise the
    *same* slider drag could push joints in opposite physical directions
    between updates as the Jacobian evolves.
    """
    _, _, Vt = np.linalg.svd(J, full_matrices=True)
    null_vec = Vt[-1]
    if prev_vec is not None and np.dot(null_vec, prev_vec) < 0:
        null_vec = -null_vec
    return null_vec


def format_solve_stats(meta, avg_total_ms):
    return (
        f"iter: {meta.iterations}   err: {meta.error:.2e}\n"
        f"build: {meta.build_time_ms:.2f} ms   opt: {meta.optimize_time_ms:.2f} ms\n"
        f"marg: {meta.marginalize_time_ms:.2f} ms   extr: {meta.extract_time_ms:.2f} ms\n"
        f"total: {meta.total_time_ms:.2f} ms   avg: {avg_total_ms:.2f} ms"
    )


def format_joint_torques(joint_torques, joint_names):
    stds = np.sqrt(np.diag(joint_torques.cov))
    return "\n".join(
        f"{name}: {mean:+7.3f} ± {std:.3f} Nm"
        for name, mean, std in zip(joint_names, joint_torques.mean, stds))


class RigidForwardSimApp:
    def __init__(self, server: viser.ViserServer):
        self.server = server
        viz_utils.setup_default_lighting(server)

        self.urdf = load_urdf()
        server.scene.add_frame("/robot", show_axes=False)
        self.viser_urdf = ViserUrdf(server, self.urdf, root_node_name="/robot")

        self.joint_names = self.viser_urdf.get_actuated_joint_names()
        self.num_joints = len(self.joint_names)
        self.limits = self.viser_urdf.get_actuated_joint_limits()
        self.joint_lo = np.array([self.limits[name][0] for name in self.joint_names])
        self.joint_hi = np.array([self.limits[name][1] for name in self.joint_names])

        # See tendon_forward_sim.py's docstring on _solve_lock: viser
        # dispatches GUI callbacks from a thread pool, and the solver isn't
        # safe to call concurrently on the same instance.
        self._solve_lock = threading.RLock()
        self._solve_times = []
        self.solver = None       # built by _rebuild_solver below
        self.joint_axes = None   # local-frame joint axes, for the plotter
        self._last_solution = None

        # See _set_sliders/tendon_forward_sim.py's docstring on the same
        # pattern: setting several sliders as one atomic update suppresses
        # the per-slider solve_and_update() cascade each .value set would
        # otherwise trigger, so only one solve happens against the fully-
        # updated state.
        self._suppress_slider_solve = False

        # Sign-continuity anchor for null_space_vector, and the null-space
        # slider's own last-seen value (see _null_space_step).
        self._null_space_prev_vec = None
        self._null_space_prev_value = 0.0

        self._build_gui()
        self.plotter = RigidRobotPlotter(
            server, joint_axes=None, prefix="/rigid_robot_uncertainty")
        self._rebuild_solver()

    def _build_gui(self):
        server = self.server

        self.sliders = []
        with server.gui.add_folder("Joints"):
            for name, value in zip(self.joint_names, home_config(self.joint_names)):
                lo, hi = self.limits[name]
                slider = server.gui.add_slider(
                    name, min=lo, max=hi, step=JOINT_STEP, initial_value=value)
                slider.on_update(lambda _: self.solve_and_update())
                self.sliders.append(slider)

            reset_joints = server.gui.add_button("Reset to home")

        with server.gui.add_folder("Null Space"):
            self.null_space_slider = server.gui.add_slider(
                "null space", min=NULL_SPACE_MIN, max=NULL_SPACE_MAX,
                step=NULL_SPACE_STEP, initial_value=0.0,
                hint="Redundant self-motion DOF: drag to move the joints (e.g. swing "
                     "the elbow) while an intermediate tip-holding solve keeps the tip "
                     "in place")
            reset_null_space = server.gui.add_button("Center null space")

            self.null_space_slider.on_update(lambda _: self._null_space_step())

            @reset_null_space.on_click
            def _(_):
                self._suppress_slider_solve = True
                try:
                    self.null_space_slider.value = 0.0
                finally:
                    self._suppress_slider_solve = False
                self._null_space_prev_value = 0.0

        with server.gui.add_folder("Uncertainty (sigma)"):
            self.joint_sigma_slider = server.gui.add_slider(
                "joint value sigma", min=JOINT_SIGMA_MIN, max=JOINT_SIGMA_MAX,
                step=JOINT_SIGMA_STEP, initial_value=JOINT_SIGMA_INITIAL,
                hint="Uncertainty on the commanded joint-value prior (rad)")
            self.joint_sigma_slider.on_update(lambda _: self.solve_and_update())

            # Offset calibration sigma is baked into the model's factor graph
            # at construction (it's a fixed property of the robot, like
            # K_inv for a rod), not a per-solve input -- changing it rebuilds
            # the solver, same as _reset_solver in the other apps.
            self.offset_sigma_rot_slider = server.gui.add_slider(
                "offset sigma (rot, rad)", min=OFFSET_SIGMA_ROT_MIN, max=OFFSET_SIGMA_ROT_MAX,
                step=OFFSET_SIGMA_ROT_STEP, initial_value=DEFAULT_OFFSET_SIGMA_ROT,
                hint="Per-joint calibration uncertainty: assembly tolerance, backlash, compliance")
            self.offset_sigma_pos_slider = server.gui.add_slider(
                "offset sigma (pos, m)", min=OFFSET_SIGMA_POS_MIN, max=OFFSET_SIGMA_POS_MAX,
                step=OFFSET_SIGMA_POS_STEP, initial_value=DEFAULT_OFFSET_SIGMA_POS)
            self.offset_sigma_rot_slider.on_update(lambda _: self._rebuild_solver())
            self.offset_sigma_pos_slider.on_update(lambda _: self._rebuild_solver())

        with server.gui.add_folder("Wrench Sensing"):
            server.gui.add_markdown(
                "Always on: a tip-wrench variable is part of every solve, with a "
                "prior from the force/moment sliders below -- see the Joint Torques "
                "folder for the resulting posterior over each joint's generalized force.")

            self.force_sliders = [
                server.gui.add_slider(label, min=FORCE_MIN, max=FORCE_MAX,
                                       step=FORCE_STEP, initial_value=0.0)
                for label in ("fx", "fy", "fz")
            ]
            self.moment_sliders = [
                server.gui.add_slider(label, min=MOMENT_MIN, max=MOMENT_MAX,
                                       step=MOMENT_STEP, initial_value=0.0)
                for label in ("mx", "my", "mz")
            ]
            reset_wrench = server.gui.add_button("Reset wrench")
            self.tip_force_sigma_slider = server.gui.add_slider(
                "tip force sigma", min=TIP_FORCE_SIGMA_MIN, max=TIP_FORCE_SIGMA_MAX,
                step=TIP_FORCE_SIGMA_STEP, initial_value=TIP_FORCE_SIGMA_INITIAL)
            self.tip_moment_sigma_slider = server.gui.add_slider(
                "tip moment sigma", min=TIP_MOMENT_SIGMA_MIN, max=TIP_MOMENT_SIGMA_MAX,
                step=TIP_MOMENT_SIGMA_STEP, initial_value=TIP_MOMENT_SIGMA_INITIAL)

            wrench_controls = (
                self.force_sliders + self.moment_sliders
                + [self.tip_force_sigma_slider, self.tip_moment_sigma_slider])
            for control in wrench_controls:
                control.on_update(lambda _: self.solve_and_update())

            @reset_wrench.on_click
            def _(_):
                for slider in self.force_sliders + self.moment_sliders:
                    slider.value = 0.0

        with server.gui.add_folder("Tip Pose Prior"):
            self.tip_pose_sigma_rot_slider = server.gui.add_slider(
                "tip pose sigma (rot, rad)", min=TIP_POSE_SIGMA_ROT_MIN, max=TIP_POSE_SIGMA_ROT_MAX,
                step=TIP_POSE_SIGMA_ROT_STEP, initial_value=TIP_POSE_SIGMA_ROT_INITIAL)
            self.tip_pose_sigma_pos_slider = server.gui.add_slider(
                "tip pose sigma (pos, m)", min=TIP_POSE_SIGMA_POS_MIN, max=TIP_POSE_SIGMA_POS_MAX,
                step=TIP_POSE_SIGMA_POS_STEP, initial_value=TIP_POSE_SIGMA_POS_INITIAL)

            for control in (self.tip_pose_sigma_rot_slider, self.tip_pose_sigma_pos_slider):
                control.on_update(lambda _: self.solve_and_update())

        with server.gui.add_folder("Tip Pose"):
            self.tip_position_readout = server.gui.add_text(
                "tip position", initial_value="", disabled=True)
            self.solve_stats_readout = server.gui.add_text(
                "solve stats", initial_value="", disabled=True)

        with server.gui.add_folder("Joint Torques (posterior)"):
            self.joint_torques_readout = server.gui.add_text(
                "torques", initial_value="", disabled=True, multiline=True)

        @reset_joints.on_click
        def _(_):
            for slider, value in zip(self.sliders, home_config(self.joint_names)):
                slider.value = value

        self.tip_frame = server.scene.add_frame(
            "/tip_frame", axes_length=0.06, axes_radius=0.003, show_axes=True)

        # Draggable target for the tip pose. Dragging it solves an
        # intermediate tip-pose-prior problem (_joints_for_tip_target) just
        # to translate "move the tip here" into joint values -- that
        # solve's own marginals are never shown. The joint sliders are then
        # updated to that result and _set_sliders triggers the one real
        # display solve (solve_and_update), which never sees a tip-pose
        # prior at all. See solve_and_update's docstring for why.
        self._gizmo_dragging = False
        self.tip_pose_gizmo = server.scene.add_transform_controls(
            "/tip_pose_target", scale=0.1)

        @self.tip_pose_gizmo.on_update
        def _(event):
            if event.phase == "start":
                self._gizmo_dragging = True
            elif event.phase == "update":
                # Must hold _solve_lock for this whole sequence, same as
                # every other path that touches self.solver: viser
                # dispatches GUI callbacks from a thread pool, and rapid
                # drag events firing solve() concurrently on the same
                # solver instance corrupts its internal (warm-start) state.
                with self._solve_lock:
                    target = viz_utils.position_wxyz_to_pose(
                        self.tip_pose_gizmo.position, self.tip_pose_gizmo.wxyz)
                    joint_mean = np.array([s.value for s in self.sliders])
                    joint_values = self._joints_for_tip_target(
                        joint_mean, target,
                        self.tip_pose_sigma_rot_slider.value, self.tip_pose_sigma_pos_slider.value)
                    self._set_sliders(zip(self.sliders, joint_values))
            elif event.phase == "end":
                self._gizmo_dragging = False
                self._reposition_tip_pose_gizmo(
                    viz_utils.position_wxyz_to_pose(self.tip_frame.position, self.tip_frame.wxyz))

    def _reposition_tip_pose_gizmo(self, tip_pose_mean):
        # Snaps the gizmo to the just-solved tip pose after every solve that
        # didn't come from actively dragging it -- so it starts from
        # wherever the arm actually is if you pick it up later, and a
        # completed drag settles onto the actual solved tip rather than the
        # imprecise dragged point.
        if self._gizmo_dragging:
            return
        self.tip_pose_gizmo.position = tip_pose_mean[:3, 3]
        self.tip_pose_gizmo.wxyz = viz_utils.pose_to_wxyz(tip_pose_mean)

    def _set_sliders(self, slider_value_pairs):
        # Sets several sliders as one atomic update: suppresses the
        # per-slider solve_and_update() cascade that setting .value would
        # otherwise trigger (each change fires its on_update callback
        # synchronously), so only one solve happens against the fully-
        # updated state -- same rationale as tendon_forward_sim.py's
        # _set_sliders.
        self._suppress_slider_solve = True
        try:
            for slider, value in slider_value_pairs:
                slider.value = value
        finally:
            self._suppress_slider_solve = False
        self.solve_and_update()

    def _joints_for_tip_target(self, joint_mean, tip_pose_target, sigma_rot, sigma_pos):
        # Intermediate, non-display solve: translates "put the tip here"
        # into joint values. Always self.solver -- never a separate
        # instance: RigidRobotSolver warm-starts itself from whatever it
        # last solved, so switching solver objects between calls means
        # whichever one you come back to has a stale warm start relative
        # to the *other* one's recent history, and can need dozens of
        # iterations to recover. Kept on the one solver, this solve starts
        # right next to the truth (self.solver's warm start is always
        # fresh -- see solve_and_update) and converges in very few. The
        # tip-pose prior added here never contaminates what's displayed:
        # solve_and_update always does one more, final solve with no
        # tip-pose prior, and that's what _last_solution/the marginals
        # shown actually come from.
        joint_cov = (self.joint_sigma_slider.value ** 2) * np.eye(self.num_joints)
        sigma = np.concatenate([np.full(3, sigma_rot), np.full(3, sigma_pos)])
        tip_pose_prior = bendier.Pose3Gaussian(tip_pose_target, np.diag(sigma ** 2))
        placeholder_wrench = bendier.Vector6Gaussian(np.zeros(6), np.eye(6))
        solution = self.solver.solve(
            bendier.VectorXGaussian(joint_mean, joint_cov), placeholder_wrench, None, tip_pose_prior)
        return solution.marginals.joints.mean

    def _null_space_step(self):
        # Relative control: the slider's absolute value has no fixed
        # physical meaning (the null direction shifts as joint values
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

            # self._last_solution.marginals.J_tip_joints is already clean --
            # solve_and_update never includes a tip-pose prior (see its
            # docstring), so nothing here entangles T's dependence on Q
            # with an external anchor.
            q0 = self._last_solution.marginals.joints.mean
            p0 = self._last_solution.marginals.tip_pose.mean  # the pose to hold fixed
            J = self._last_solution.marginals.J_tip_joints
            null_vec = null_space_vector(J, self._null_space_prev_vec)
            self._null_space_prev_vec = null_vec

            q1 = np.clip(q0 + delta * null_vec, self.joint_lo, self.joint_hi)

            # Inverse solve: joint prior at q1, holding the *original* tip
            # pose p0 fixed. self.solver's warm start is already sitting
            # right at q0 (from the last display solve -- see
            # solve_and_update, which always runs on this same instance),
            # and q1 is only a small step away from that, so this converges
            # in very few iterations rather than needing a large
            # from-scratch correction.
            corrected_joints = self._joints_for_tip_target(
                q1, p0, NULL_SPACE_HOLD_SIGMA_ROT, NULL_SPACE_HOLD_SIGMA_POS)

            # Final display update: the corrected joints are already
            # (almost) exactly on the tip-preserving manifold, and
            # self.solver is now warm right at them, so solve_and_update's
            # own solve converges immediately and the tip shouldn't visibly
            # move.
            self._set_sliders(zip(self.sliders, corrected_joints))

    def _rebuild_solver(self):
        # Cheap to rebuild from scratch -- see _reset_solver in the other
        # apps for the same "throw it away rather than mutate" rationale.
        # Needed whenever a graph-structure-level setting changes (currently
        # just the offset sigma), since that's baked in at RigidRobotModel
        # construction, not per-solve.
        with self._solve_lock:
            joint_specs = build_joint_specs(
                self.urdf,
                sigma_offset_rot=self.offset_sigma_rot_slider.value,
                sigma_offset_pos=self.offset_sigma_pos_slider.value)
            self.joint_axes = [np.array(spec.axis) for spec in joint_specs]
            self.plotter.joint_axes = self.joint_axes

            base_calibration = build_base_calibration()
            tip_offset_calibration = build_tip_offset_calibration(self.urdf)

            config = bendier.RigidRobotSolverConfig(
                joint_specs, base_calibration, tip_offset_calibration,
                enable_wrench_sensing=True)
            self.solver = bendier.RigidRobotSolver(config)
            self._solve_times = []
            self._null_space_prev_vec = None

            self.solve_and_update()

    def solve_and_update(self):
        # The one and only display path -- always the last thing solved on
        # self.solver every time, and always a plain solve on the current
        # joint-value slider prior (plus the wrench prior), never a
        # tip-pose prior: we can only truly measure/command joint values on
        # a real robot, so the displayed uncertainty should always reflect
        # that, not an artificial "we directly know the tip pose" claim.
        # Anything that wants to reach a tip target (dragging the gizmo,
        # the null-space slider's hold) first solves that intermediately
        # via _joints_for_tip_target -- on this *same* solver instance, so
        # it stays warm-started at the truth throughout -- then updates the
        # joint sliders and lets *this* function run the final, clean solve
        # and display the result the same way every time, regardless of
        # which control triggered it.
        with self._solve_lock:
            if self._suppress_slider_solve:
                return

            joint_mean = np.array([s.value for s in self.sliders])
            joint_cov = (self.joint_sigma_slider.value ** 2) * np.eye(self.num_joints)
            joint_prior = bendier.VectorXGaussian(joint_mean, joint_cov)

            moment_mean = np.array([s.value for s in self.moment_sliders])
            force_mean = np.array([s.value for s in self.force_sliders])
            wrench_mean = np.concatenate([moment_mean, force_mean])
            wrench_sigma = np.concatenate([
                np.full(3, self.tip_moment_sigma_slider.value),
                np.full(3, self.tip_force_sigma_slider.value),
            ])
            tip_wrench_prior = bendier.Vector6Gaussian(wrench_mean, np.diag(wrench_sigma ** 2))

            solution = self.solver.solve(joint_prior, tip_wrench_prior)
            self._last_solution = solution

            # No slider-resync-to-posterior needed: this solve never
            # includes a tip-pose prior, so the posterior joint values
            # always equal the slider-commanded means (up to solver
            # tolerance) -- nothing here ever pulls them away.

            # The mesh is still driven by yourdfpy's own FK (via
            # ViserUrdf.update_cfg) rather than reading solved link poses
            # directly -- simpler, and mathematically the same forward
            # kinematics either way.
            self.viser_urdf.update_cfg(solution.marginals.joints.mean)

            tip = solution.marginals.tip_pose
            self.tip_frame.position = tip.mean[:3, 3]
            self.tip_frame.wxyz = viz_utils.pose_to_wxyz(tip.mean)
            self._reposition_tip_pose_gizmo(tip.mean)
            self.plotter.update(solution)

            self._solve_times.append(solution.meta.total_time_ms)
            self.tip_position_readout.value = (
                f"[{tip.mean[0, 3]:.4f}, {tip.mean[1, 3]:.4f}, {tip.mean[2, 3]:.4f}] m")
            self.solve_stats_readout.value = format_solve_stats(
                solution.meta, np.mean(self._solve_times))
            self.joint_torques_readout.value = format_joint_torques(
                solution.marginals.joint_torques, self.joint_names)


def main():
    server = viser.ViserServer()
    print("Open the URL above in a browser, then drag the sliders in the GUI panel.")
    RigidForwardSimApp(server)
    server.sleep_forever()


if __name__ == "__main__":
    main()
