"""Shared helpers for the viser-based interactive plotters.

Mirrors the role of `bendier.plotting.utils` for the pyvista plotters, but
geared toward viser's persistent-scene-graph model instead of pyvista's
per-frame vtkTransform pattern:

- Meshes (tubes, discs, ellipsoids) are built once and their `.vertices` /
  `.position` / `.wxyz` / `.scale` are mutated in place on every solve --
  this is the expensive path (topology data), so it's worth not resending.
- Lines and frames are cheap enough (no separate topology payload) that we
  just re-add them by name each solve; viser treats that as an update.
"""

import numpy as np
from scipy.spatial.transform import Rotation

frame_arrow_colors = ["red", "green", "blue"]

# RGB equivalents of the named pyvista/VTK colors used throughout
# bendier.plotting, looked up via `pv.Color(name).int_rgb` -- kept here so
# every viser plotter references the same values instead of re-approximating
# them (some, like the "cadmium" family, aren't standard CSS colors).
SILVER = (192, 192, 192)
ULTRAMARINE = (18, 10, 143)
DEEP_CADMIUM_RED = (227, 23, 13)
CADMIUM_LEMON = (255, 227, 3)
CORNFLOWER_BLUE = (100, 149, 237)
DARK_ORCHID = (153, 50, 204)
DEEP_PINK = (255, 20, 147)
RED = (255, 0, 0)
GREEN = (0, 128, 0)


def pose_to_wxyz(pose_mean: np.ndarray) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(pose_mean[:3, :3]).as_quat()
    return np.array([w, x, y, z])


def setup_default_lighting(server, environment_intensity=0.6, sun_intensity=3.0):
    server.scene.configure_environment_map(
        "city", background=False, environment_intensity=environment_intensity)
    server.scene.add_light_directional(
        "/lights/sun", color=(255, 255, 255), intensity=sun_intensity,
        cast_shadow=True, position=(0.2, 0.1, 0.4))
    server.scene.add_light_ambient(
        "/lights/ambient", color=(255, 255, 255), intensity=0.3)


# --- Procedural tube mesh (rod backbone) --------------------------------
#
# Cross-section rings are oriented using each node's own material frame
# (columns 0/1 of its rotation matrix), so the tube twists along with the
# actual solved rod orientation rather than a generic Frenet frame. Face
# topology only depends on node count, which is fixed for a given config,
# so it's computed once by the caller and reused.

def tube_faces(num_rings: int, sides: int) -> np.ndarray:
    faces = []
    for i in range(num_rings - 1):
        base0, base1 = i * sides, (i + 1) * sides
        for j in range(sides):
            j2 = (j + 1) % sides
            a, b = base0 + j, base0 + j2
            c, d = base1 + j2, base1 + j
            faces.append((a, b, c))
            faces.append((a, c, d))
    return np.array(faces, dtype=np.uint32)


def tube_vertices(poses, radius: float, sides: int) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    cos_a, sin_a = np.cos(angles)[None, :, None], np.sin(angles)[None, :, None]

    positions = np.array([T[:3, 3] for T in poses])
    x_axes = np.array([T[:3, 0] for T in poses])
    y_axes = np.array([T[:3, 1] for T in poses])

    rings = (
        positions[:, None, :]
        + radius * cos_a * x_axes[:, None, :]
        + radius * sin_a * y_axes[:, None, :]
    )
    return rings.reshape(-1, 3).astype(np.float32)


# --- Discs / plates ---------------------------------------------------
#
# These are genuine straight cylinders (fixed radius/height, one pose each)
# -- viser's native add_cylinder (height along local z, matching our pose
# convention exactly) handles this directly, so there's no reason to hand-
# build vertex/face buffers the way the tube does. add_cylinder / the
# returned CylinderHandle also cover normals, material, and shading
# correctly on their own. Only the tube backbone still needs custom mesh
# code, since it bends/twists along the rod and has no matching primitive.

def add_disc(scene, name, pose, radius, half_width, color, **kwargs):
    return scene.add_cylinder(
        name, radius=radius, height=2 * half_width, color=color,
        position=pose[:3, 3], wxyz=pose_to_wxyz(pose), **kwargs)


def update_disc(handle, pose):
    handle.position = pose[:3, 3]
    handle.wxyz = pose_to_wxyz(pose)


# --- Covariance ellipsoids ------------------------------------------------
#
# Built from a unit `add_icosphere` primitive, oriented/scaled per-axis via
# eigendecomposition of the covariance -- avoids hand-building sphere mesh
# geometry, and lets us update just `.position` / `.wxyz` / `.scale` (all
# tiny payloads) rather than re-sending mesh data every solve.

def ellipsoid_wxyz_scale(cov: np.ndarray, scale: float = 1.0, num_sigma: float = 2.0):
    eigvals, eigvecs = np.linalg.eigh(cov)
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 0] *= -1
    radii = num_sigma * np.sqrt(np.maximum(eigvals, 1e-12)) * scale
    x, y, z, w = Rotation.from_matrix(eigvecs).as_quat()
    return np.array([w, x, y, z]), tuple(radii)


def add_ellipsoid(scene, name, position, cov, color, opacity=0.3, scale=1.0, num_sigma=2.0):
    wxyz, radii = ellipsoid_wxyz_scale(cov, scale=scale, num_sigma=num_sigma)
    handle = scene.add_icosphere(
        name, radius=1.0, position=position, wxyz=wxyz, color=color, opacity=opacity)
    handle.scale = radii
    return handle


def update_ellipsoid(handle, position, cov, scale=1.0, num_sigma=2.0):
    wxyz, radii = ellipsoid_wxyz_scale(cov, scale=scale, num_sigma=num_sigma)
    handle.position = position
    handle.wxyz = wxyz
    handle.scale = radii


# --- Vector arrows (force/moment) -----------------------------------------
#
# viser's native add_arrows (cone-tipped, same points=(N,2,3) convention as
# add_line_segments) -- re-added by name each solve, same as everywhere else
# that doesn't bother with in-place updates for cheap-payload scene nodes.
# Shaft/head thickness is a fixed size the caller picks (appropriate to that
# robot's own scale), not derived from the vector's length -- scaling
# thickness with magnitude made bigger forces render as fatter arrows, which
# reads as wrong (thickness isn't the thing encoding magnitude, length is).

def set_vector_arrow(scene, name, origin, vec, scale, color, shaft_radius=0.004):
    end = origin + vec * scale
    return scene.add_arrows(
        name, points=np.array([[origin, end]]), colors=color,
        shaft_radius=shaft_radius, head_radius=shaft_radius * 3.0, head_length=shaft_radius * 6.0)
