"""Interactive forward-mechanics viewer for the tendon-driven robot.

Drag the tension/tip-force sliders in the browser GUI and the rod shape
updates live. This is a viser-based experiment (see README's viser note) --
it does not yet support saving a video the way the pyvista plotters do.

Rendering lives in bendier.viser_plotting.ViserTendonRobotPlotter, shared
with the cosserat and parallel-robot viser apps.

Run with:
    python python/apps/tendon_forward_sim.py
then open the printed http://localhost:8080 URL in a browser.
"""

import os
import sys
import time

import numpy as np
import viser

import bendier
from bendier.viser_plotting import ViserTendonRobotPlotter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "tendon_robot"))
from config import get_config, get_tendon_input  # noqa: E402

TENSION_MIN, TENSION_MAX, TENSION_STEP = 0.0, 5.0, 0.05
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


class TendonForwardSimApp:
    def __init__(self, server: viser.ViserServer):
        self.server = server
        self.solver = bendier.TendonRobotSolver(get_config())
        self.plotter = ViserTendonRobotPlotter(
            server, plot_tip_force=True, plot_backbone_ellipsoids=True)

        self.num_tendons = len(get_tendon_input().functions)

        self._build_gui()
        self.solve_and_update()

    def _build_gui(self):
        server = self.server

        with server.gui.add_folder("Tendon Tensions"):
            self.tension_sliders = [
                server.gui.add_slider(
                    f"tendon {i}", min=TENSION_MIN, max=TENSION_MAX,
                    step=TENSION_STEP, initial_value=0.0)
                for i in range(self.num_tendons)
            ]
            reset_tensions = server.gui.add_button("Reset tensions")

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
            self.solve_time_readout = server.gui.add_text(
                "solve time", initial_value="", disabled=True)

        sigma_sliders = [self.tension_sigma_slider, self.force_sigma_slider]
        for slider in self.tension_sliders + self.force_sliders + sigma_sliders:
            slider.on_update(lambda _: self.solve_and_update())

        @reset_tensions.on_click
        def _(_):
            for slider in self.tension_sliders:
                slider.value = 0.0

        @reset_wrench.on_click
        def _(_):
            for slider in self.force_sliders:
                slider.value = 0.0

    def solve_and_update(self):
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

        solution = self.solver.solve(tensions, tip_wrench, None)
        self.plotter.update(solution)

        tip_position = solution.marginals.rod.states[-1].pose.mean[:3, 3]
        self.tip_position_readout.value = (
            f"[{tip_position[0]:.4f}, {tip_position[1]:.4f}, {tip_position[2]:.4f}] m")
        self.solve_time_readout.value = f"{solution.meta.total_time_ms:.2f} ms"


def main():
    server = viser.ViserServer()
    print("Open the URL above in a browser, then drag the sliders in the GUI panel.")
    TendonForwardSimApp(server)

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    main()
