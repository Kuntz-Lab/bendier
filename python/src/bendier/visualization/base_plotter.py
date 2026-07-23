"""Shared lifecycle/GUI plumbing for every per-robot viser plotter.

Before this, each of the 4 per-robot plotters (Cosserat rod, tendon,
parallel, rigid) reimplemented server ownership and close() slightly
differently, and every *app* (not the plotters at all) hand-duplicated an
identical format_solve_stats() function plus the tracking/readout code
around it -- with a bug common to all 4 copies: the readout's multi-line
string was never given multiline=True, so it rendered squashed/hard to
read in the sidebar (confirmed by contrast with rigid_robot/app.py's
joint_torques_readout, which does set it and displays correctly).
"""

import numpy as np
import viser

from . import utils


def format_solve_stats(meta, avg_total_ms):
    """One stat per line with aligned labels -- the old format packed two
    heavily-abbreviated stats per line (e.g. "marg: 0.12 ms   extr: 0.03
    ms"), which read as noise more than information in a narrow sidebar
    column, especially since it was never rendered as actual multiple
    lines to begin with (see module docstring).
    """
    return (
        f"iterations   {meta.iterations}\n"
        f"error        {meta.error:.3e}\n"
        f"build        {meta.build_time_ms:7.2f} ms\n"
        f"optimize     {meta.optimize_time_ms:7.2f} ms\n"
        f"marginalize  {meta.marginalize_time_ms:7.2f} ms\n"
        f"extract      {meta.extract_time_ms:7.2f} ms\n"
        f"total        {meta.total_time_ms:7.2f} ms\n"
        f"avg total    {avg_total_ms:7.2f} ms"
    )


class BasePlotter:
    """Common base for the per-robot viser plotters.

    Owns the server (creating one if none is given, so a standalone script
    can construct a plotter with no separate viser setup of its own -- the
    same convenience a pyvista plotter owning its own window gives you),
    sets up default lighting once, and provides a standardized solve-stats
    readout via update_solve_stats().

    Subclasses call update_solve_stats(solution.meta) themselves at the end
    of their own update() rather than this class imposing a single update()
    signature -- each robot's update() legitimately takes different extra
    arguments (e.g. tip_force_gt, p_desired), so forcing one template-method
    signature here would fight that more than it would help.
    """

    def __init__(self, server=None, port=8080, show_solve_stats=True):
        if server is None:
            server = viser.ViserServer(port=port)
            print("Open the URL above in a browser to watch.")
        self.server = server
        utils.setup_default_lighting(server)

        self._solve_times = []
        self.solve_stats_readout = None
        if show_solve_stats:
            with server.gui.add_folder("Solve Stats"):
                self.solve_stats_readout = server.gui.add_text(
                    "stats", initial_value="", disabled=True, multiline=True)

    def update_solve_stats(self, meta):
        """Call once per update() with the just-returned solution.meta.
        No-op if this plotter was constructed with show_solve_stats=False.
        """
        if self.solve_stats_readout is None:
            return
        self._solve_times.append(meta.total_time_ms)
        self.solve_stats_readout.value = format_solve_stats(meta, float(np.mean(self._solve_times)))

    def reset_solve_stats(self):
        """Call from an app's own solver-reset flow so the running average
        doesn't mix pre- and post-reset solve times."""
        self._solve_times = []

    def close(self):
        pass
