import threading

import numpy as np
import viser

import bendier
from bendier.visualization import CosseratRodPlotter

# Running this file directly (`python app.py`) puts its own directory first
# on sys.path automatically, so config.py -- right next to this file -- is
# importable with no manual path setup.
from config import get_config

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


class CosseratRodApp:
    def __init__(self, server: viser.ViserServer):
        self.server = server
        self.solver = bendier.CosseratRodSolver(get_config())
        self.plotter = CosseratRodPlotter(
            server, plot_wrenches=True, plot_backbone_ellipsoids=True)
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
        self._suppress_slider_solve = False

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
            self.status_readout = server.gui.add_text(
                "status", initial_value="ok", disabled=True)
            reset_solver = server.gui.add_button("Reset solver")

        all_sliders = (
            self.moment_sliders + self.force_sliders
            + self.moment_sigma_sliders + self.force_sigma_sliders)
        for slider in all_sliders:
            slider.on_update(lambda _: self.solve_and_update())

        @reset_wrench.on_click
        def _(_):
            self._set_sliders((s, 0.0) for s in self.moment_sliders + self.force_sliders)

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
            self.solver = bendier.CosseratRodSolver(get_config())
            self.plotter.reset_solve_stats()
            self._set_sliders(
                [(s, 0.0) for s in self.moment_sliders + self.force_sliders]
                + [(s, MOMENT_SIGMA_INITIAL) for s in self.moment_sigma_sliders]
                + [(s, FORCE_SIGMA_INITIAL) for s in self.force_sigma_sliders])

    def solve_and_update(self):
        with self._solve_lock:
            if self._suppress_slider_solve:
                return

            moment_mean = np.array([s.value for s in self.moment_sliders])
            force_mean = np.array([s.value for s in self.force_sliders])
            tip_wrench_mean = np.concatenate([moment_mean, force_mean])

            sigma = np.array(
                [s.value for s in self.moment_sigma_sliders]
                + [s.value for s in self.force_sigma_sliders])
            wrench_cov = np.diag(sigma ** 2)

            tip_wrench = bendier.Vector6Gaussian(tip_wrench_mean, wrench_cov)

            try:
                solution = self.solver.solve(tip_wrench=tip_wrench)
            except Exception as e:
                print(f"[cosserat_rod/app] solve() failed, resetting solver: {e}")
                self.solver = bendier.CosseratRodSolver(get_config())
                self.status_readout.value = f"solve failed ({type(e).__name__}) -- solver reset"
                return

            self.plotter.update(solution)

            tip_position = solution.marginals.states[-1].pose.mean[:3, 3]
            self.tip_position_readout.value = (
                f"[{tip_position[0]:.4f}, {tip_position[1]:.4f}, {tip_position[2]:.4f}] m")
            self.status_readout.value = "ok"


def main():
    server = viser.ViserServer()
    print("Open the URL above in a browser, then drag the sliders in the GUI panel.")
    CosseratRodApp(server)
    server.sleep_forever()


if __name__ == "__main__":
    main()
