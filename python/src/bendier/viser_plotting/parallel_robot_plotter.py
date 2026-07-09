"""viser equivalent of bendier.plotting.parallel_robot_plotter.

Reuses ViserCosseratRodMeshManager once per rod (rod count from the
solution, not hardcoded) plus a platform disc, mirroring the pyvista
ParallelRobotPlotter's constructor options and method names.
"""

import itertools

import numpy as np

from . import utils
from .cosserat_rod_plotter import ViserCosseratRodMeshManager

ROD_COLORS = [
    (102, 51, 153), (227, 23, 13), (255, 97, 3),
    (255, 176, 15), (46, 139, 87), (65, 105, 225),
]
PLATFORM_SIDES = 24
PLATFORM_LEG_SIDES = 8


class ViserParallelRobotPlotter:
    def __init__(self,
                 server,
                 platform_z_offset=0.0,
                 plot_rod_wrenches=True,
                 plot_tip_force=True,
                 plot_base_wrenches=False,
                 plot_backbone_frames=False,
                 plot_backbone_ellipsoids=True):

        self.server = server
        utils.setup_default_lighting(server)

        self.rod_managers = None

        self.moment_scale = 0.2
        self.force_scale = 0.3
        self.platform_z_offset = platform_z_offset
        self.plot_tip_force = plot_tip_force
        self._plot_rod_wrenches = plot_rod_wrenches
        self._plot_base_wrenches = plot_base_wrenches
        self._plot_backbone_frames = plot_backbone_frames
        self._plot_backbone_ellipsoids = plot_backbone_ellipsoids

        self.platform_mesh = None
        self.platform_leg_mesh = None
        self.platform_joint = None
        self.platform_ellipsoid = None
        self.tip_force_ellipsoid = None

        base_vertices = utils.disc_vertices([np.eye(4)], 0.2, 0.01, PLATFORM_SIDES)
        base_faces = utils.disc_faces(1, PLATFORM_SIDES)
        self.server.scene.add_mesh_simple(
            "/base_plate", vertices=base_vertices, faces=base_faces,
            color=utils.SILVER, flat_shading=True, side="double", opacity=0.5)

    def _ensure_rod_managers(self, num_rods):
        if self.rod_managers is not None:
            return

        self.rod_managers = []
        colors = itertools.cycle(ROD_COLORS)
        for i in range(num_rods):
            self.rod_managers.append(ViserCosseratRodMeshManager(
                self.server.scene,
                prefix=f"/rods/{i}",
                plot_base_plate=False,
                plot_wrenches=self._plot_rod_wrenches,
                plot_base_wrench=self._plot_base_wrenches,
                plot_backbone_frames=self._plot_backbone_frames,
                plot_backbone_ellipsoids=self._plot_backbone_ellipsoids,
                backbone_radius=0.005,
                moment_scale=self.moment_scale,
                force_scale=self.force_scale,
                cartesian_frame_scale=0.025,
                rod_opacity=0.7,
                rod_color=next(colors),
            ))

    def update_platform(self, solution):
        pose = solution.platform_pose.mean
        p, R = pose[:3, 3], pose[:3, :3]
        local_z = R[:, 2]

        # `platform_pose` tracks a reference frame (roughly the mean of the
        # rod tip attachments); the visible plate sits offset from it along
        # that frame's own z-axis by platform_z_offset -- matching pyvista's
        # version, which bakes the same offset into the mesh's local points
        # before applying the pose transform.
        plate_pose = pose.copy()
        plate_pose[:3, 3] = p + local_z * self.platform_z_offset

        vertices = utils.disc_vertices([plate_pose], 0.15, 0.005, PLATFORM_SIDES)
        if self.platform_mesh is None:
            faces = utils.disc_faces(1, PLATFORM_SIDES)
            self.platform_mesh = self.server.scene.add_mesh_simple(
                "/platform/plate", vertices=vertices, faces=faces,
                color=utils.SILVER, flat_shading=True, side="double", opacity=0.85)
            self.platform_mesh.cast_shadow = True
            self.platform_mesh.receive_shadow = True
        else:
            self.platform_mesh.vertices = vertices

        # Thin leg connecting the tracked reference frame down to the plate.
        leg_pose = pose.copy()
        leg_pose[:3, 3] = p + local_z * (self.platform_z_offset / 2.0)
        leg_vertices = utils.disc_vertices(
            [leg_pose], 0.005, abs(self.platform_z_offset) / 2.0, PLATFORM_LEG_SIDES)
        if self.platform_leg_mesh is None:
            leg_faces = utils.disc_faces(1, PLATFORM_LEG_SIDES)
            self.platform_leg_mesh = self.server.scene.add_mesh_simple(
                "/platform/leg", vertices=leg_vertices, faces=leg_faces,
                color=utils.SILVER, flat_shading=True, side="double")
        else:
            self.platform_leg_mesh.vertices = leg_vertices

        if self.platform_joint is None:
            self.platform_joint = self.server.scene.add_icosphere(
                "/platform/joint", radius=0.01, position=p, color=utils.SILVER)
        else:
            self.platform_joint.position = p

        cov = R @ (solution.platform_pose.cov[3:, 3:] @ R.T)
        if self.platform_ellipsoid is None:
            self.platform_ellipsoid = utils.add_ellipsoid(
                self.server.scene, "/platform_ellipsoid", p, cov, color=utils.DEEP_CADMIUM_RED, opacity=0.2)
        else:
            utils.update_ellipsoid(self.platform_ellipsoid, p, cov)

    def update_tip_force(self, solution, tip_force_gt=None):
        if not self.plot_tip_force:
            return

        p = solution.platform_pose.mean[:3, 3]
        f = solution.platform_wrench.mean[3:]
        cov = solution.platform_wrench.cov[3:, 3:]

        utils.set_vector_line(
            self.server.scene, "/platform_tip_force", p, f, self.force_scale, color=utils.DARK_ORCHID)

        if self.tip_force_ellipsoid is None:
            self.tip_force_ellipsoid = utils.add_ellipsoid(
                self.server.scene, "/platform_tip_force_ellipsoid",
                p + f * self.force_scale, cov, color=utils.CADMIUM_LEMON, opacity=0.4, scale=self.force_scale)
        else:
            utils.update_ellipsoid(
                self.tip_force_ellipsoid, p + f * self.force_scale, cov, scale=self.force_scale)

        if tip_force_gt is not None:
            utils.set_vector_line(
                self.server.scene, "/platform_tip_force_gt", p, tip_force_gt,
                self.force_scale, color=utils.GREEN)

    def update(self, solution, tip_force_gt=None):
        self._ensure_rod_managers(len(solution.marginals.rods))
        for i, manager in enumerate(self.rod_managers):
            manager.update(solution.marginals.rods[i])

        self.update_platform(solution.marginals)
        self.update_tip_force(solution.marginals, tip_force_gt)

    def close(self):
        pass
