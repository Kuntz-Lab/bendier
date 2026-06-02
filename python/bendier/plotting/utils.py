import os
from pathlib import Path
import shutil

import matplotlib.pyplot as plt

import numpy as np
import pyvista as pv


frame_arrow_colors = ["red", "green", "blue"]


def setup_plt(width=3.0, height=5.0, grid=False):

    os.makedirs("figures", exist_ok=True)

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
        "pdf.fonttype": 42,  # embed fonts in PDF
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",  # math text compatible with Times
        "mathtext.rm": "stix",
        "lines.markersize": 4
    })


def get_tube_from_points(points, radius):
    spline = pv.Spline(points, n_points=200)
    tube = spline.tube(radius=radius)

    return tube


def get_tube_from_poses(poses, radius):
    points = np.array([T[:3, 3] for T in poses])
    return get_tube_from_points(points, radius)


def get_ellipsoid_transform(center, cov, scale=1.0, num_sigma=2.0):
    eigvals, eigvecs = np.linalg.eigh(cov)
    one_sigma = np.sqrt(np.maximum(eigvals, 1e-12)) * scale
    radii = num_sigma * one_sigma

    A = eigvecs @ np.diag(radii)
    T = np.eye(4)
    T[:3, :3] = A
    T[:3,  3] = center

    return T


def get_arrow(length=1.0, direction=None, shaft_scale=1.0):
    if direction is None:
        direction = np.array([1, 0, 0])

    shaft_prescale = 0.05

    arrow = pv.Arrow(
        start=np.zeros(3),
        direction=direction,
        scale=length,
        tip_resolution=20,
        shaft_resolution=20,
        shaft_radius=shaft_prescale *  shaft_scale,
        tip_radius=2 * shaft_prescale * shaft_scale,
        tip_length=2 * shaft_prescale * shaft_scale
    )

    return arrow


def get_axes_frame(length=1.0):
    return [
        get_arrow(length=length, direction=np.eye(3)[:,0]),
        get_arrow(length=length, direction=np.eye(3)[:,1]),
        get_arrow(length=length, direction=np.eye(3)[:,2])
    ]


def get_arrow_transform(p, vec, scale=1.0):
    length = np.linalg.norm(vec) * scale
    if length < 1e-12:
        dir = np.array([1.0, 0.0, 0.0])
    else:
        dir = vec / np.linalg.norm(vec)

    x_axis = np.array([1.0, 0.0, 0.0])
    v = np.cross(x_axis, dir)
    c = np.dot(x_axis, dir)
    if np.linalg.norm(v) < 1e-12:
        R = np.eye(3) if c > 0 else -np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1 / (1 + c))

    # Scale along x for magnitude, then rotate x vector to the target vector 
    T = np.eye(4)
    T[:3, :3] = R @ np.diag([length, 1.0, 1.0])
    T[:3, 3] = p

    return T


class PlotterBase:
    def __init__(self,
                 save_frames_dir_name=None, 
                 single_plot_mode=False,
                 plot_rviz_coords=False,
                 camera_focal_point=None,
                 camera_azimuth=15,
                 camera_elevation=20,
                 camera_distance=0.6):
        
        self.save_frames_dir_name = save_frames_dir_name
        self.single_plot_mode = single_plot_mode
        self.plot_rviz_coords = plot_rviz_coords

        if camera_focal_point is None:
            self.camera_focal_point = np.zeros(3)
        else:
            self.camera_focal_point = camera_focal_point
        
        self.camera_azimuth = camera_azimuth
        self.camera_elevation = camera_elevation
        self.camera_distance = camera_distance

        if self.save_frames_dir_name:
            self.frames_path = Path("videos") / "frames" / self.save_frames_dir_name
            shutil.rmtree(self.frames_path, ignore_errors=True)
            self.frames_path.mkdir(parents=True, exist_ok=True)

        self.window_size = (4000, 4000)
        self.plotter = pv.Plotter(window_size=self.window_size, off_screen=save_frames_dir_name)
        self.frame = 0
        self.solve_time_ms_history = []

        self.init_scene()

    def init_scene(self):
        p = self.camera_focal_point
        a = np.deg2rad(self.camera_azimuth)
        e = np.deg2rad(self.camera_elevation)
        d = self.camera_distance

        x = p[0] + d * np.cos(e) * np.cos(a)
        y = p[1] + d * np.cos(e) * np.sin(a)
        z = p[2] + d * np.sin(e)

        self.plotter.camera.position = (x, y, z)
        self.plotter.camera.focal_point = self.camera_focal_point

        if self.plot_rviz_coords:
            self.plotter.add_axes()
        
        # self.plotter.enable_depth_peeling(10)
        self.plotter.enable_anti_aliasing()
    
    def update(self, solution):

        if self.frame == 0:
            show_plot = not self.save_frames_dir_name
            if show_plot:
                interactive_update = not self.single_plot_mode
                self.plotter.show(auto_close=False, interactive_update=interactive_update)

        self.solve_time_ms_history.append(solution.meta.total_time_ms)

        text = (
            f"iter: {solution.meta.iterations:3d}, "
            f"error: {solution.meta.error:3.2e}, "
            f"build: {solution.meta.build_time_ms:3.2f} ms, "
            f"optimize: {solution.meta.optimize_time_ms:3.2f} ms, "
            f"marginalize: {solution.meta.marginalize_time_ms:3.2f} ms, "
            f"extract: {solution.meta.extract_time_ms:3.2f} ms, "
            f"total: {solution.meta.total_time_ms:3.2f} ms, "
            f"avg: {np.mean(self.solve_time_ms_history):3.2f} ms"
        )
        
        self.plotter.add_text(text, position='upper_right', font_size=14, font="courier", name="solve_time")

        self.plotter.render()

        if self.save_frames_dir_name:
            self.plotter.screenshot(self.frames_path / f"{self.frame}.png", window_size=self.window_size)
        
        self.frame += 1
