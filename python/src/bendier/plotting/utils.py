import os

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


frame_arrow_colors = ["red", "green", "blue"]


def setup_plt(width=3.0, height=5.0, grid=False):
    os.makedirs("output/figures", exist_ok=True)
    plt.rcParams.update({
        "figure.figsize": (width, height),
        "font.family": "STIXGeneral",
        "font.size": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": grid,
        "grid.alpha": 0.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",
        "mathtext.rm": "stix",
        "lines.markersize": 4,
    })


def get_tube_from_points(points, radius):
    spline = pv.Spline(points, n_points=200)
    return spline.tube(radius=radius)


def get_tube_from_poses(poses, radius):
    points = np.array([T[:3, 3] for T in poses])
    return get_tube_from_points(points, radius)


def get_ellipsoid_transform(center, cov, scale=1.0, num_sigma=2.0):
    eigvals, eigvecs = np.linalg.eigh(cov)
    radii = num_sigma * np.sqrt(np.maximum(eigvals, 1e-12)) * scale
    A = eigvecs @ np.diag(radii)
    T = np.eye(4)
    T[:3, :3] = A
    T[:3, 3] = center
    return T


def get_arrow(length=1.0, direction=None, shaft_scale=1.0):
    if direction is None:
        direction = np.array([1, 0, 0])
    shaft_prescale = 0.05
    return pv.Arrow(
        start=np.zeros(3),
        direction=direction,
        scale=length,
        tip_resolution=20,
        shaft_resolution=20,
        shaft_radius=shaft_prescale * shaft_scale,
        tip_radius=2 * shaft_prescale * shaft_scale,
        tip_length=2 * shaft_prescale * shaft_scale,
    )


def get_axes_frame(length=1.0):
    return [get_arrow(length=length, direction=np.eye(3)[:, i]) for i in range(3)]


def get_arrow_transform(p, vec, scale=1.0):
    length = np.linalg.norm(vec) * scale
    if length < 1e-12:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = vec / np.linalg.norm(vec)

    x_axis = np.array([1.0, 0.0, 0.0])
    v = np.cross(x_axis, direction)
    c = np.dot(x_axis, direction)
    if np.linalg.norm(v) < 1e-12:
        R = np.eye(3) if c > 0 else -np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1 / (1 + c))

    T = np.eye(4)
    T[:3, :3] = R @ np.diag([length, 1.0, 1.0])
    T[:3, 3] = p
    return T


class PlotterBase:
    """Thin wrapper around pv.Plotter that owns the render/record loop.

    The window is always shown live and interactive (you can rotate/pan/zoom
    while the simulation runs). Passing save_movie additionally records every
    frame to an mp4 alongside the live view.

    Usage
    -----
        plotter = PlotterBase(save_movie="out.mp4", ...)  # save_movie optional
        for sol in solutions:
            plotter.update(sol)
        plotter.close()   # finalises the movie file, if any
    """

    def __init__(
        self,
        save_movie: str | None = None,
        frame_rate: float = 30.0,
        window_size: tuple = (1440, 1360),
        pause_each_frame: bool = False,
        show_axes: bool = False,
        camera_focal_point=None,
        camera_azimuth: float = 15.0,
        camera_elevation: float = 20.0,
        camera_distance: float = 0.6,
    ):
        self.save_movie = save_movie
        self.frame_rate = frame_rate
        self._pause_each_frame = pause_each_frame
        self._advance = True
        self.frame = 0
        self._solve_times: list[float] = []
        self._text_actor = None

        if save_movie:
            movie_dir = os.path.dirname(save_movie)
            if movie_dir:
                os.makedirs(movie_dir, exist_ok=True)

        self.plotter = pv.Plotter(window_size=window_size)
        self.plotter.enable_anti_aliasing()

        if show_axes:
            self.plotter.add_axes()

        p = np.zeros(3) if camera_focal_point is None else np.asarray(camera_focal_point, dtype=float)
        a = np.deg2rad(camera_azimuth)
        e = np.deg2rad(camera_elevation)
        d = camera_distance
        self.plotter.camera.position = (
            p[0] + d * np.cos(e) * np.cos(a),
            p[1] + d * np.cos(e) * np.sin(a),
            p[2] + d * np.sin(e),
        )
        self.plotter.camera.focal_point = p.tolist()

    def _on_first_frame(self):
        if self.save_movie:
            self.plotter.open_movie(self.save_movie, framerate=int(round(self.frame_rate)))
        if self._pause_each_frame:
            self.plotter.add_key_event("space", lambda: setattr(self, "_advance", True))
        self.plotter.show(auto_close=False, interactive_update=True)

    def update(self, solution):
        if self.frame == 0:
            self._on_first_frame()

        meta = solution.meta
        self._solve_times.append(meta.total_time_ms)
        text = (
            f"iter: {meta.iterations:3d}   err: {meta.error:8.2e}   "
            f"build: {meta.build_time_ms:5.2f} ms   opt: {meta.optimize_time_ms:5.2f} ms   "
            f"marg: {meta.marginalize_time_ms:5.2f} ms   extr: {meta.extract_time_ms:5.2f} ms   "
            f"total: {meta.total_time_ms:6.2f} ms   avg: {np.mean(self._solve_times):6.2f} ms"
        )

        if self._text_actor is None:
            self._text_actor = self.plotter.add_text(
                text, position="upper_right", font_size=12, font="courier"
            )
        else:
            self._text_actor.set_text("upper_right", text)

        # write_frame() (and update()) call iren.process_events() internally,
        # which is what keeps the window responsive to mouse/keyboard input
        # in this single-threaded loop -- no separate render thread needed.
        if self.save_movie:
            self.plotter.write_frame()
        else:
            self.plotter.update()

        if self._pause_each_frame:
            self._advance = False
            while not self._advance:
                self.plotter.update()

        self.frame += 1

    def close(self):
        self.plotter.close()
