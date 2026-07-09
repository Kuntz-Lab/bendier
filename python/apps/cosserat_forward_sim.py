"""Interactive forward-mechanics viewer for a single Cosserat rod.

Drag the tip-wrench sliders in the browser GUI and the rod shape updates
live. Unlike the tendon/parallel apps, this one exposes the full 6D tip
wrench (moment + force) with an independent uncertainty sigma per
component, since a bare rod has no actuation input to focus on instead.

Rendering lives in bendier.viser_plotting.ViserCosseratRodPlotter, shared
with the tendon and parallel-robot viser apps.

Run with:
    python python/apps/cosserat_forward_sim.py
then open the printed http://localhost:8080 URL in a browser.
"""

import os
import sys
import time

import numpy as np
import viser

import bendier
from bendier.viser_plotting import ViserCosseratRodPlotter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "cosserat"))
from config import get_config  # noqa: E402

MOMENT_MIN, MOMENT_MAX, MOMENT_STEP = -0.5, 0.5, 0.005
FORCE_MIN, FORCE_MAX, FORCE_STEP = -1.0, 1.0, 0.01

# Sigma sliders: how tightly each tip-wrench component's prior pins the
# solve to its mean. Small sigma is near-deterministic (the classic
# forward-mechanics case); larger sigma lets the estimate drift away from
# the slider value in response to other factors in the graph -- which is
# the actual Bayesian behavior on display, not just a cosmetic knob. Each
# of the 6 wrench components gets its own sigma slider.
MOMENT_SIGMA_MIN, MOMENT_SIGMA_MAX, MOMENT_SIGMA_STEP = 0.0001, 0.2, 0.0001
MOMENT_SIGMA_INITIAL = 0.001
FORCE_SIGMA_MIN, FORCE_SIGMA_MAX, FORCE_SIGMA_STEP = 0.0001, 0.5, 0.0001
FORCE_SIGMA_INITIAL = 0.001


class CosseratForwardSimApp:
    def __init__(self, server: viser.ViserServer):
        self.server = server
        self.solver = bendier.CosseratRodSolver(get_config())
        self.plotter = ViserCosseratRodPlotter(
            server, plot_wrenches=True, plot_backbone_ellipsoids=True)

        self._build_gui()
        self.solve_and_update()

    def _build_gui(self):
        server = self.server

        with server.gui.add_folder("Tip Wrench"):
            self.moment_sliders = [
                server.gui.add_slider(
                    label, min=MOMENT_MIN, max=MOMENT_MAX,
                    step=MOMENT_STEP, initial_value=0.0)
                for label in ("mx", "my", "mz")
            ]
            self.force_sliders = [
                server.gui.add_slider(
                    label, min=FORCE_MIN, max=FORCE_MAX,
                    step=FORCE_STEP, initial_value=0.0)
                for label in ("fx", "fy", "fz")
            ]
            reset_wrench = server.gui.add_button("Reset wrench")

        with server.gui.add_folder("Uncertainty (sigma)"):
            self.moment_sigma_sliders = [
                server.gui.add_slider(
                    f"{label} sigma", min=MOMENT_SIGMA_MIN, max=MOMENT_SIGMA_MAX,
                    step=MOMENT_SIGMA_STEP, initial_value=MOMENT_SIGMA_INITIAL)
                for label in ("mx", "my", "mz")
            ]
            self.force_sigma_sliders = [
                server.gui.add_slider(
                    f"{label} sigma", min=FORCE_SIGMA_MIN, max=FORCE_SIGMA_MAX,
                    step=FORCE_SIGMA_STEP, initial_value=FORCE_SIGMA_INITIAL)
                for label in ("fx", "fy", "fz")
            ]

        with server.gui.add_folder("Solution"):
            self.tip_position_readout = server.gui.add_text(
                "tip position", initial_value="", disabled=True)
            self.solve_time_readout = server.gui.add_text(
                "solve time", initial_value="", disabled=True)

        all_sliders = (
            self.moment_sliders + self.force_sliders
            + self.moment_sigma_sliders + self.force_sigma_sliders)
        for slider in all_sliders:
            slider.on_update(lambda _: self.solve_and_update())

        @reset_wrench.on_click
        def _(_):
            for slider in self.moment_sliders + self.force_sliders:
                slider.value = 0.0

    def solve_and_update(self):
        moment_mean = np.array([s.value for s in self.moment_sliders])
        force_mean = np.array([s.value for s in self.force_sliders])
        tip_wrench_mean = np.concatenate([moment_mean, force_mean])

        sigma = np.array(
            [s.value for s in self.moment_sigma_sliders]
            + [s.value for s in self.force_sigma_sliders])
        wrench_cov = np.diag(sigma ** 2)

        tip_wrench = bendier.Vector6Gaussian(tip_wrench_mean, wrench_cov)

        solution = self.solver.solve(tip_wrench=tip_wrench)
        self.plotter.update(solution)

        tip_position = solution.marginals.states[-1].pose.mean[:3, 3]
        self.tip_position_readout.value = (
            f"[{tip_position[0]:.4f}, {tip_position[1]:.4f}, {tip_position[2]:.4f}] m")
        self.solve_time_readout.value = f"{solution.meta.total_time_ms:.2f} ms"


def main():
    server = viser.ViserServer()
    print("Open the URL above in a browser, then drag the sliders in the GUI panel.")
    CosseratForwardSimApp(server)

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    main()
