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
    load_urdf, home_config, build_joint_specs, build_base_calibration,
    DEFAULT_OFFSET_SIGMA_ROT, DEFAULT_OFFSET_SIGMA_POS)

JOINT_STEP = 0.01

JOINT_SIGMA_MIN, JOINT_SIGMA_MAX, JOINT_SIGMA_STEP = 0.0001, 0.3, 0.0001
JOINT_SIGMA_INITIAL = 0.01

OFFSET_SIGMA_ROT_MIN, OFFSET_SIGMA_ROT_MAX, OFFSET_SIGMA_ROT_STEP = 1e-5, 0.05, 1e-5
OFFSET_SIGMA_POS_MIN, OFFSET_SIGMA_POS_MAX, OFFSET_SIGMA_POS_STEP = 1e-5, 0.02, 1e-5

FORCE_MIN, FORCE_MAX, FORCE_STEP = -20.0, 20.0, 0.1
# Moments aren't exposed as sliders on the tip-wrench prior -- pinned to zero
# with a small fixed sigma so they stay negligible without cluttering the
# GUI, same rationale as the other apps' tip-wrench priors.
TIP_MOMENT_SIGMA_FIXED = 0.01
TIP_FORCE_SIGMA_MIN, TIP_FORCE_SIGMA_MAX, TIP_FORCE_SIGMA_STEP = 0.001, 2.0, 0.001
TIP_FORCE_SIGMA_INITIAL = 0.05


def format_solve_stats(meta, avg_total_ms):
    return (
        f"iter: {meta.iterations}   err: {meta.error:.2e}\n"
        f"build: {meta.build_time_ms:.2f} ms   opt: {meta.optimize_time_ms:.2f} ms\n"
        f"marg: {meta.marginalize_time_ms:.2f} ms   extr: {meta.extract_time_ms:.2f} ms\n"
        f"total: {meta.total_time_ms:.2f} ms   avg: {avg_total_ms:.2f} ms"
    )


def format_joint_torques(joint_torques, joint_names):
    if joint_torques is None:
        return "(enable wrench sensing above to see this)"

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

        # See tendon_forward_sim.py's docstring on _solve_lock: viser
        # dispatches GUI callbacks from a thread pool, and the solver isn't
        # safe to call concurrently on the same instance.
        self._solve_lock = threading.RLock()
        self._solve_times = []
        self.solver = None       # built by _rebuild_solver below
        self.joint_axes = None   # local-frame joint axes, for the plotter

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
            self.wrench_sensing_checkbox = server.gui.add_checkbox(
                "enable wrench sensing", initial_value=False,
                hint="Adds a tip-wrench variable to the factor graph, with a prior "
                     "from the force sliders below -- see the Joint Torques folder "
                     "for the resulting posterior over each joint's generalized force")
            self.wrench_sensing_checkbox.on_update(lambda _: self._rebuild_solver())

            self.force_sliders = [
                server.gui.add_slider(label, min=FORCE_MIN, max=FORCE_MAX,
                                       step=FORCE_STEP, initial_value=0.0)
                for label in ("fx", "fy", "fz")
            ]
            self.tip_force_sigma_slider = server.gui.add_slider(
                "tip force sigma", min=TIP_FORCE_SIGMA_MIN, max=TIP_FORCE_SIGMA_MAX,
                step=TIP_FORCE_SIGMA_STEP, initial_value=TIP_FORCE_SIGMA_INITIAL)

            for control in self.force_sliders + [self.tip_force_sigma_slider]:
                control.on_update(lambda _: self.solve_and_update())

        with server.gui.add_folder("Tip Pose"):
            self.tip_position_readout = server.gui.add_text(
                "tip position", initial_value="", disabled=True)
            self.solve_stats_readout = server.gui.add_text(
                "solve stats", initial_value="", disabled=True)

        with server.gui.add_folder("Joint Torques (posterior)"):
            self.joint_torques_readout = server.gui.add_text(
                "torques", initial_value=format_joint_torques(None, self.joint_names),
                disabled=True, multiline=True)

        @reset_joints.on_click
        def _(_):
            for slider, value in zip(self.sliders, home_config(self.joint_names)):
                slider.value = value

        self.tip_frame = server.scene.add_frame(
            "/tip_frame", axes_length=0.06, axes_radius=0.003, show_axes=True)

    def _rebuild_solver(self):
        # Cheap to rebuild from scratch -- see _reset_solver in the other
        # apps for the same "throw it away rather than mutate" rationale.
        # Needed whenever a graph-structure-level setting changes (offset
        # sigma, or whether wrench sensing is enabled at all), since those
        # are baked in at RigidRobotModel construction, not per-solve.
        with self._solve_lock:
            joint_specs = build_joint_specs(
                self.urdf,
                sigma_offset_rot=self.offset_sigma_rot_slider.value,
                sigma_offset_pos=self.offset_sigma_pos_slider.value)
            self.joint_axes = [np.array(spec.axis) for spec in joint_specs]
            self.plotter.joint_axes = self.joint_axes

            base_calibration = build_base_calibration()

            config = bendier.RigidRobotSolverConfig(
                joint_specs, base_calibration,
                enable_wrench_sensing=self.wrench_sensing_checkbox.value)
            self.solver = bendier.RigidRobotSolver(config)
            self._solve_times = []
            self.solve_and_update()

    def solve_and_update(self):
        with self._solve_lock:
            joint_mean = np.array([s.value for s in self.sliders])
            joint_cov = (self.joint_sigma_slider.value ** 2) * np.eye(self.num_joints)
            joint_prior = bendier.VectorXGaussian(joint_mean, joint_cov)

            tip_wrench_prior = None
            if self.wrench_sensing_checkbox.value:
                force_mean = np.array([s.value for s in self.force_sliders])
                wrench_mean = np.concatenate([np.zeros(3), force_mean])
                wrench_sigma = np.concatenate([
                    np.full(3, TIP_MOMENT_SIGMA_FIXED),
                    np.full(3, self.tip_force_sigma_slider.value),
                ])
                tip_wrench_prior = bendier.Vector6Gaussian(wrench_mean, np.diag(wrench_sigma ** 2))

            solution = self.solver.solve(joint_prior, tip_wrench_prior)

            # The mesh is still driven by yourdfpy's own FK (via
            # ViserUrdf.update_cfg) for simplicity -- it agrees with the
            # solver's mean pose exactly (both are the same forward
            # kinematics), so only the *uncertainty*/*wrench* visualization
            # actually needs the factor-graph solve.
            self.viser_urdf.update_cfg(joint_mean)

            tip = solution.marginals.links[-1].pose
            self.tip_frame.position = tip.mean[:3, 3]
            self.tip_frame.wxyz = viz_utils.pose_to_wxyz(tip.mean)
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
