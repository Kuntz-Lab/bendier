"""Interactive forward-mechanics viewer for the parallel (Stewart-style) robot.

Drag the rod-length and platform-force sliders in the browser GUI and the
platform pose updates live. Mirrors cosserat_forward_sim.py's/
tendon_forward_sim.py's structure (same sigma-slider convention, moments
pinned small).

Rendering lives in bendier.viser_plotting.ViserParallelRobotPlotter, shared
with the cosserat and tendon-robot viser apps.

Run with:
    python python/apps/parallel_forward_sim.py
then open the printed http://localhost:8080 URL in a browser.
"""

import os
import sys
import time

import numpy as np
import viser

import bendier
from bendier.viser_plotting import ViserParallelRobotPlotter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "parallel_robot"))
from config import get_config, get_base_poses, platform_z_offset  # noqa: E402

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


class ParallelForwardSimApp:
    def __init__(self, server: viser.ViserServer):
        self.server = server
        self.solver = bendier.ParallelRobotSolver(get_config())
        self.plotter = ViserParallelRobotPlotter(
            server, platform_z_offset=platform_z_offset,
            plot_rod_wrenches=False, plot_tip_force=True)

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
            self.solve_time_readout = server.gui.add_text(
                "solve time", initial_value="", disabled=True)

        sigma_sliders = [self.rod_lengths_sigma_slider, self.force_sigma_slider]
        for slider in self.rod_length_sliders + self.force_sliders + sigma_sliders:
            slider.on_update(lambda _: self.solve_and_update())

        @reset_lengths.on_click
        def _(_):
            for slider in self.rod_length_sliders:
                slider.value = ROD_LENGTH_INITIAL

        @reset_wrench.on_click
        def _(_):
            for slider in self.force_sliders:
                slider.value = 0.0

    def solve_and_update(self):
        rod_lengths = np.array([s.value for s in self.rod_length_sliders])
        force_mean = np.array([s.value for s in self.force_sliders])
        wrench_mean = np.concatenate([np.zeros(3), force_mean])

        wrench_sigma = np.concatenate([
            np.full(3, MOMENT_SIGMA_FIXED),
            np.full(3, self.force_sigma_slider.value),
        ])
        wrench_cov = np.diag(wrench_sigma ** 2)
        wrench = bendier.Vector6Gaussian(wrench_mean, wrench_cov)

        solution = self.solver.solve(
            rod_lengths, self.rod_lengths_sigma_slider.value, wrench, None)
        self.plotter.update(solution)

        platform_position = solution.marginals.platform_pose.mean[:3, 3]
        self.platform_position_readout.value = (
            f"[{platform_position[0]:.4f}, {platform_position[1]:.4f}, {platform_position[2]:.4f}] m")
        self.solve_time_readout.value = f"{solution.meta.total_time_ms:.2f} ms"


def main():
    server = viser.ViserServer()
    print("Open the URL above in a browser, then drag the sliders in the GUI panel.")
    ParallelForwardSimApp(server)

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    main()
