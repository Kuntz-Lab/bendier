"""viser equivalent of bendier.plotting.tendon_robot_plotter.

Reuses ViserCosseratRodMeshManager for the backbone tube and adds the
tendon-routing discs and tendon lines on top, mirroring the pyvista
TendonRobotPlotter's constructor options and method names.
"""

import itertools

import numpy as np

from . import utils
from .cosserat_rod_plotter import ViserCosseratRodMeshManager

TENDON_COLORS = [
    (220, 20, 60), (34, 139, 34), (65, 105, 225),
    (186, 85, 211), (218, 165, 32), (255, 20, 147),
]
DISC_SIDES = 16


class ViserTendonRobotPlotter:
    def __init__(self,
                 server,
                 plot_rod_wrenches=False,
                 plot_tip_force=True,
                 plot_base_wrenches=False,
                 plot_backbone_frames=False,
                 plot_backbone_ellipsoids=True):

        self.server = server
        utils.setup_default_lighting(server)

        self.plot_tip_force = plot_tip_force

        self.rod_manager = ViserCosseratRodMeshManager(
            server.scene,
            plot_backbone_ellipsoids=plot_backbone_ellipsoids,
            plot_wrenches=plot_rod_wrenches,
            plot_base_wrench=plot_base_wrenches,
            plot_backbone_frames=plot_backbone_frames,
            skip_backbone_ellipsoids=4,
            backbone_radius=0.001,
            cartesian_frame_scale=0.01,
            force_scale=0.05,
            moment_scale=0.2,
        )

        self.disc_mesh = None
        self.tip_force_ellipsoid = None

    def update_tendons(self, solution):
        tendon_config = solution.marginals.tendon_config
        num_tendons = tendon_config.num_tendons
        disc_pose_idx = tendon_config.disc_pose_idx
        hole_locations = tendon_config.hole_locations  # [disc][tendon]
        states = solution.marginals.rod.states

        colors = list(itertools.islice(itertools.cycle(TENDON_COLORS), num_tendons))

        # Tendons are taut, straight cords between routing holes -- straight
        # line segments, not an interpolated curve.
        for tendon_idx in range(num_tendons):
            points = np.array([
                states[pose_idx].pose.mean[:3, :3] @ hole_locations[disc_idx][tendon_idx]
                + states[pose_idx].pose.mean[:3, 3]
                for disc_idx, pose_idx in enumerate(disc_pose_idx)
            ])
            segments = np.stack([points[:-1], points[1:]], axis=1)
            self.server.scene.add_line_segments(
                f"/tendons/{tendon_idx}", points=segments,
                colors=colors[tendon_idx], line_width=2.5)

    def update_discs(self, solution):
        tendon_config = solution.marginals.tendon_config
        disc_pose_idx = tendon_config.disc_pose_idx
        states = solution.marginals.rod.states

        disc_poses = [states[i].pose.mean for i in disc_pose_idx]
        radius = 1.3 * tendon_config.routing_radius
        half_width = 0.15 * tendon_config.routing_radius

        vertices = utils.disc_vertices(disc_poses, radius, half_width, DISC_SIDES)
        if self.disc_mesh is None:
            faces = utils.disc_faces(len(disc_poses), DISC_SIDES)
            self.disc_mesh = self.server.scene.add_mesh_simple(
                "/rod/discs", vertices=vertices, faces=faces,
                color=utils.CORNFLOWER_BLUE, opacity=0.7, flat_shading=True, side="double")
            self.disc_mesh.cast_shadow = True
            self.disc_mesh.receive_shadow = True
        else:
            self.disc_mesh.vertices = vertices

    def update_tip_force(self, solution, tip_force_gt=None):
        if not self.plot_tip_force:
            return

        force_scale = 0.3

        p = solution.marginals.rod.states[-1].pose.mean[:3, 3]
        f = solution.marginals.external_wrenches[-1].mean[3:]
        cov = solution.marginals.external_wrenches[-1].cov[3:, 3:]

        utils.set_vector_line(
            self.server.scene, "/rod/tip_force", p, f, force_scale, color=utils.DARK_ORCHID)

        if self.tip_force_ellipsoid is None:
            self.tip_force_ellipsoid = utils.add_ellipsoid(
                self.server.scene, "/rod/tip_force_ellipsoid",
                p + f * force_scale, cov, color=utils.CADMIUM_LEMON, opacity=0.4, scale=force_scale)
        else:
            utils.update_ellipsoid(
                self.tip_force_ellipsoid, p + f * force_scale, cov, scale=force_scale)

        if tip_force_gt is not None:
            utils.set_vector_line(
                self.server.scene, "/rod/tip_force_gt", p, tip_force_gt, force_scale, color=utils.GREEN)

    def update_p_desired(self, p):
        self.server.scene.add_icosphere(
            "/rod/p_desired", radius=0.002, position=p, color=utils.RED)

    def update(self, solution, p_desired=None, tip_force_gt=None):
        self.rod_manager.update(solution.marginals.rod)
        self.update_tendons(solution)
        self.update_discs(solution)
        self.update_tip_force(solution, tip_force_gt)

        if p_desired is not None:
            self.update_p_desired(p_desired)

    def close(self):
        pass
