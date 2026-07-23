# TODO: review this file
"""Shared helpers for the viser-based interactive plotters.

Geared toward viser's persistent-scene-graph model:

- Meshes (tubes, discs, ellipsoids) are built once and their `.vertices` /
  `.position` / `.wxyz` / `.scale` are mutated in place on every solve --
  this is the expensive path (topology data), so it's worth not resending.
- Lines and frames are cheap enough (no separate topology payload) that we
  just re-add them by name each solve; viser treats that as an update.
"""

import time

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

frame_arrow_colors = ["red", "green", "blue"]


# --- Real-time playback pacing ---------------------------------------------
#
# The batch scripts all loop solve() + plotter.update() and want the result
# to play back at roughly a fixed frame rate. A plain time.sleep(dt) after
# each iteration doesn't account for how long the solve/update themselves
# took, so it runs *slower* than the target rate (each iteration actually
# takes dt + solve/update time); dropping the sleep entirely runs as fast as
# solve() allows, often much faster than intended. FramePacer sleeps only the
# remaining time needed to hit the target, and if a frame runs over budget it
# resyncs to "now" rather than trying to burst through extra frames to catch
# up (which would look worse than just briefly running behind).

class FramePacer:
    def __init__(self, dt: float):
        self.dt = dt
        self._next_tick = time.perf_counter() + dt

    def tick(self):
        remaining = self._next_tick - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
            self._next_tick += self.dt
        else:
            self._next_tick = time.perf_counter() + self.dt

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


def pose_batch_to_wxyz(poses: np.ndarray) -> np.ndarray:
    """Vectorized pose_to_wxyz over an (N, 4, 4) stack of poses."""
    xyzw = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    return np.concatenate([xyzw[:, 3:4], xyzw[:, :3]], axis=1)


def position_wxyz_to_pose(position: np.ndarray, wxyz: np.ndarray) -> np.ndarray:
    """Inverse of pose_to_wxyz -- builds a 4x4 pose from a viser scene node's
    own .position/.wxyz (e.g. a transform-controls gizmo the user dragged)."""
    w, x, y, z = wxyz
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    pose[:3, 3] = position
    return pose


def setup_default_lighting(server, environment_intensity=0.6, sun_intensity=3.0):
    # cast_shadow is left at its default (False) -- a shadow-casting light on
    # thin/flat objects (discs, the tube) produced fine-grained shadow-acne
    # moire on their faces, which read as a mesh/normals bug but was really
    # just the shadow map; simplest fix is to not generate shadow maps at all.
    server.scene.add_light_directional(
        "/lights/sun", color=(255, 255, 255), intensity=sun_intensity,
        position=(0.2, 0.1, 0.4))
    server.scene.add_light_ambient(
        "/lights/ambient", color=(255, 255, 255), intensity=0.3)


# --- Pose interpolation (for smooth tube rendering) -----------------------
#
# Solver nodes can be sparse (a 15-node rod has 15 poses total), so placing
# one tube ring exactly at each pose looks faceted along the rod's length,
# not just around its circumference. This densifies the pose list between
# consecutive nodes before tube_vertices ever sees it: cubic Hermite on
# position, using each node's own local z-axis (the rod's tangent, per this
# codebase's pose convention) as the derivative -- scaled by chord length,
# the standard choice absent an explicit arclength parametrization -- plus
# SLERP on orientation, so the cross-section frame (and therefore twist)
# interpolates smoothly too, not just the centerline.
#
# Fully vectorized over both node-pairs and interpolation steps at once
# (one batched Rotation.from_matrix/from_quat call each, no per-pair Slerp
# object and no per-step Python-level call) -- this is called every solve,
# and the original per-pair-per-step scipy Slerp loop measured *more*
# expensive than the physics solve itself for a 50-node rod (~7ms vs
# ~1ms). Verified numerically identical to the old implementation (exact
# position match, ~1e-7 rad rotation match, floating-point noise) across
# 200 random pose sequences; ~16-34x faster.

def interpolate_poses(poses, segments_per_interval: int):
    if segments_per_interval <= 1 or len(poses) < 2:
        return list(poses)

    poses_arr = np.asarray(poses)
    T0, T1 = poses_arr[:-1], poses_arr[1:]
    p0, p1 = T0[:, :3, 3], T1[:, :3, 3]
    chord = np.linalg.norm(p1 - p0, axis=1)
    m0 = T0[:, :3, 2] * chord[:, None]
    m1 = T1[:, :3, 2] * chord[:, None]

    q0 = Rotation.from_matrix(T0[:, :3, :3]).as_quat()
    q1 = Rotation.from_matrix(T1[:, :3, :3]).as_quat()

    # Shortest-path correction: flip q1 if the pair is on opposite
    # hemispheres of the double-cover, same as scipy's Slerp does internally.
    dot = np.sum(q0 * q1, axis=1)
    q1 = np.where((dot < 0)[:, None], -q1, q1)
    theta = np.arccos(np.clip(np.abs(dot), -1.0, 1.0))

    s = np.arange(segments_per_interval) / segments_per_interval

    s2, s3 = s**2, s**3
    h00 = 2 * s3 - 3 * s2 + 1
    h10 = s3 - 2 * s2 + s
    h01 = -2 * s3 + 3 * s2
    h11 = s3 - s2

    positions = (
        h00[None, :, None] * p0[:, None, :] + h10[None, :, None] * m0[:, None, :]
        + h01[None, :, None] * p1[:, None, :] + h11[None, :, None] * m1[:, None, :]
    )

    # Vectorized SLERP: w0*q0 + w1*q1 (then renormalized) is equivalent to
    # the standard slerp formula for every (pair, step) at once. Falls back
    # to linear blending when theta ~ 0 (near-identical consecutive
    # orientations), where sin(theta) underflows and the slerp weights
    # would otherwise divide by ~zero.
    sin_theta = np.sin(theta)
    near_zero = sin_theta < 1e-8
    safe_sin_theta = np.where(near_zero, 1.0, sin_theta)

    w0 = np.sin((1 - s)[None, :] * theta[:, None]) / safe_sin_theta[:, None]
    w1 = np.sin(s[None, :] * theta[:, None]) / safe_sin_theta[:, None]
    w0 = np.where(near_zero[:, None], 1 - s[None, :], w0)
    w1 = np.where(near_zero[:, None], s[None, :], w1)

    quats = w0[:, :, None] * q0[:, None, :] + w1[:, :, None] * q1[:, None, :]
    quats /= np.linalg.norm(quats, axis=2, keepdims=True)

    num_pairs = T0.shape[0]
    rotations = Rotation.from_quat(quats.reshape(-1, 4)).as_matrix().reshape(
        num_pairs, segments_per_interval, 3, 3)

    dense = np.tile(np.eye(4), (num_pairs, segments_per_interval, 1, 1))
    dense[:, :, :3, :3] = rotations
    dense[:, :, :3, 3] = positions
    dense_list = list(dense.reshape(-1, 4, 4))
    dense_list.append(poses_arr[-1])
    return dense_list


# --- Procedural tube mesh (rod backbone) --------------------------------
#
# Cross-section rings are oriented using each node's own material frame
# (columns 0/1 of its rotation matrix), so the tube twists along with the
# actual solved rod orientation rather than a generic Frenet frame. Face
# topology only depends on ring count, which is fixed for a given config,
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
# Built from a unit icosphere, oriented/scaled per-axis via eigendecomposition
# of the covariance. Single ellipsoids (there's only ever one, e.g. the tip
# force ellipsoid) use viser's native add_icosphere and just move/reorient in
# place. Groups of ellipsoids (backbone ellipsoids, one per rod node) are
# batched into a single add_batched_meshes_simple call instead of one
# add_icosphere per node -- N separate scene nodes means N property-set
# messages over the websocket every solve; one batched node means the whole
# group updates as a single message regardless of N.

_ICOSPHERE = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
ICOSPHERE_VERTICES = _ICOSPHERE.vertices.astype(np.float32)
ICOSPHERE_FACES = _ICOSPHERE.faces.astype(np.uint32)


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
        name, radius=1.0, position=position, wxyz=wxyz, color=color, opacity=opacity,
        cast_shadow=False, receive_shadow=False)
    handle.scale = radii
    return handle


def update_ellipsoid(handle, position, cov, scale=1.0, num_sigma=2.0):
    wxyz, radii = ellipsoid_wxyz_scale(cov, scale=scale, num_sigma=num_sigma)
    handle.position = position
    handle.wxyz = wxyz
    handle.scale = radii


def ellipsoid_batch_wxyz_scale(covs: np.ndarray, scale: float = 1.0, num_sigma: float = 2.0):
    """Vectorized ellipsoid_wxyz_scale over an (N, 3, 3) stack of covariances."""
    eigvals, eigvecs = np.linalg.eigh(covs)  # (N, 3), (N, 3, 3)
    eigvecs = eigvecs.copy()
    flip = np.linalg.det(eigvecs) < 0
    eigvecs[flip, :, 0] *= -1
    radii = num_sigma * np.sqrt(np.maximum(eigvals, 1e-12)) * scale  # (N, 3)
    xyzw = Rotation.from_matrix(eigvecs).as_quat()  # (N, 4)
    wxyz = np.concatenate([xyzw[:, 3:4], xyzw[:, :3]], axis=1)
    return wxyz.astype(np.float32), radii.astype(np.float32)


def add_ellipsoid_batch(scene, name, positions, covs, color, opacity=0.3, scale=1.0, num_sigma=2.0):
    wxyz, radii = ellipsoid_batch_wxyz_scale(covs, scale=scale, num_sigma=num_sigma)
    return scene.add_batched_meshes_simple(
        name, vertices=ICOSPHERE_VERTICES, faces=ICOSPHERE_FACES,
        batched_wxyzs=wxyz, batched_positions=np.asarray(positions, dtype=np.float32),
        batched_scales=radii, batched_colors=color, opacity=opacity,
        material="toon5", cast_shadow=False, receive_shadow=False)


def update_ellipsoid_batch(handle, positions, covs, scale=1.0, num_sigma=2.0):
    wxyz, radii = ellipsoid_batch_wxyz_scale(covs, scale=scale, num_sigma=num_sigma)
    handle.batched_positions = np.asarray(positions, dtype=np.float32)
    handle.batched_wxyzs = wxyz
    handle.batched_scales = radii


# --- Batched decorative spheres (e.g. tendon routing holes) ---------------
#
# Same batching rationale as the ellipsoids above, minus the orientation
# math -- spheres are rotationally symmetric, so only position varies.

def add_sphere_batch(scene, name, positions, radius, color, opacity=1.0):
    positions = np.asarray(positions, dtype=np.float32)
    identity_wxyz = np.tile((1.0, 0.0, 0.0, 0.0), (len(positions), 1)).astype(np.float32)
    scales = np.full(len(positions), radius, dtype=np.float32)
    return scene.add_batched_meshes_simple(
        name, vertices=ICOSPHERE_VERTICES, faces=ICOSPHERE_FACES,
        batched_wxyzs=identity_wxyz, batched_positions=positions,
        batched_scales=scales, batched_colors=color, opacity=opacity,
        material="toon5", cast_shadow=False, receive_shadow=False)


def update_sphere_batch(handle, positions):
    handle.batched_positions = np.asarray(positions, dtype=np.float32)


# --- Vector arrows (force/moment) -----------------------------------------
#
# viser's native add_arrows (cone-tipped, same points=(N,2,3) convention as
# add_line_segments). Shaft/head thickness is a fixed size the caller picks
# (appropriate to that robot's own scale), not derived from the vector's
# length -- scaling thickness with magnitude made bigger forces render as
# fatter arrows, which reads as wrong (thickness isn't the thing encoding
# magnitude, length is).
#
# Callers pass back their previous handle so an arrow below min_length can be
# hidden (.visible = False) rather than left rendering a degenerate sliver --
# an arrow shorter than its own head cone is visual noise, not information.
# Below threshold, we still return the (now-hidden) handle so the caller can
# keep reusing it if the vector grows again later.

def set_vector_arrow(scene, name, origin, vec, scale, color, shaft_radius=0.004,
                      handle=None, min_length=None):
    if min_length is None:
        min_length = shaft_radius * 6.0  # matches the default head_length below

    length = np.linalg.norm(vec) * scale
    if length < min_length:
        if handle is not None:
            handle.visible = False
        return handle

    points = np.array([[origin, origin + vec * scale]], dtype=np.float32)
    if handle is None:
        return scene.add_arrows(
            name, points=points, colors=color,
            shaft_radius=shaft_radius, head_radius=shaft_radius * 3.0, head_length=shaft_radius * 6.0)

    handle.visible = True
    handle.points = points
    return handle


# --- Batched vector arrows (e.g. per-node wrench arrows along a rod) ------
#
# add_arrows already accepts points of shape (N, 2, 3) for N independent
# arrows in one call -- set_vector_arrow above only ever uses N=1, so a
# caller looping it once per rod node (e.g. one arrow per wrench estimate)
# was creating one scene node per node instead of using that native batching.
# Below-threshold vectors are dropped from the batch for that frame instead
# of needing per-arrow .visible toggling, which a single shared scene node
# can't do individually anyway.

def set_vector_arrows_batch(scene, name, origins, vecs, scale, color, shaft_radius=0.004,
                             handle=None, min_length=None):
    if min_length is None:
        min_length = shaft_radius * 6.0

    origins = np.asarray(origins)
    ends = origins + np.asarray(vecs) * scale
    keep = np.linalg.norm(ends - origins, axis=1) >= min_length

    if not np.any(keep):
        if handle is not None:
            handle.visible = False
        return handle

    points = np.stack([origins[keep], ends[keep]], axis=1).astype(np.float32)

    if handle is None:
        return scene.add_arrows(
            name, points=points, colors=color,
            shaft_radius=shaft_radius, head_radius=shaft_radius * 3.0, head_length=shaft_radius * 6.0)

    handle.visible = True
    handle.points = points
    return handle
