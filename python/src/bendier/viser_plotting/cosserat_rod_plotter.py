"""viser equivalent of bendier.plotting.cosserat_rod_plotter.

Mirrors the pyvista CosseratRodMeshManager/CosseratRodPlotter split and
constructor options as closely as viser's primitives allow, so this is the
one place backbone-tube/wrench/ellipsoid rendering lives -- reused by the
tendon and parallel-robot viser plotters exactly like the pyvista version
reuses CosseratRodMeshManager.
"""

import numpy as np
import viser

from . import utils

BACKBONE_SIDES = 12
END_PLATE_SIDES = 32
END_PLATE_MATERIAL = "toon5"


class ViserCosseratRodMeshManager:
    def __init__(self,
                 scene,
                 prefix="/rod",
                 plot_base_plate=True,
                 plot_tip_plate=False,
                 plot_wrenches=True,
                 plot_base_wrench=False,
                 plot_backbone_frames=False,
                 plot_backbone_ellipsoids=True,
                 skip_backbone_ellipsoids=1,
                 backbone_radius=0.005,
                 moment_scale=0.2,
                 force_scale=0.1,
                 base_plate_size=0.1,
                 cartesian_frame_scale=0.01,
                 rod_color=utils.ULTRAMARINE,
                 rod_opacity=0.8,
                 arrow_shaft_radius=None):

        self.scene = scene
        self.prefix = prefix

        self.plot_base_plate = plot_base_plate
        self.plot_tip_plate = plot_tip_plate
        self.plot_wrenches = plot_wrenches
        self.plot_base_wrench = plot_base_wrench
        self.plot_backbone_frames = plot_backbone_frames
        self.plot_backbone_ellipsoids = plot_backbone_ellipsoids
        self.skip_backbone_ellipsoids = skip_backbone_ellipsoids

        self.rod_color = rod_color
        self.rod_opacity = rod_opacity
        self.backbone_radius = backbone_radius
        self.moment_scale = moment_scale
        self.force_scale = force_scale
        self.base_plate_size = base_plate_size
        self.cartesian_frame_scale = cartesian_frame_scale
        # Sized off the rod's own thickness by default, so arrows look
        # proportionate to the rod without a separate knob to tune per robot.
        self.arrow_shaft_radius = (
            arrow_shaft_radius if arrow_shaft_radius is not None else backbone_radius * 0.8)

        self.backbone_mesh = None
        self.base_plate_mesh = None
        self.tip_plate_mesh = None
        self.backbone_ellipsoid_handles = []
        self.wrench_moment_ellipsoid_handles = []
        self.wrench_force_ellipsoid_handles = []

    def update_base_plate(self, solution):
        if not self.plot_base_plate:
            return

        pose = solution.states[0].pose.mean
        half_thick = (self.base_plate_size / 10.0) / 2.0
        if self.base_plate_mesh is None:
            self.base_plate_mesh = utils.add_disc(
                self.scene, f"{self.prefix}/base_plate", pose,
                self.base_plate_size / 2.0, half_thick, utils.SILVER,
                radial_segments=END_PLATE_SIDES, material=END_PLATE_MATERIAL)
        else:
            utils.update_disc(self.base_plate_mesh, pose)

    def update_tip_plate(self, solution):
        if not self.plot_tip_plate:
            return

        pose = solution.states[-1].pose.mean
        half_thick = (self.base_plate_size / 10.0) / 2.0
        if self.tip_plate_mesh is None:
            self.tip_plate_mesh = utils.add_disc(
                self.scene, f"{self.prefix}/tip_plate", pose,
                self.base_plate_size / 2.0, half_thick, utils.SILVER,
                radial_segments=END_PLATE_SIDES, material=END_PLATE_MATERIAL)
        else:
            utils.update_disc(self.tip_plate_mesh, pose)

    def update_rod_tube(self, solution):
        poses = [state.pose.mean for state in solution.states]
        vertices = utils.tube_vertices(poses, self.backbone_radius, BACKBONE_SIDES)

        if self.backbone_mesh is None:
            faces = utils.tube_faces(len(poses), BACKBONE_SIDES)
            self.backbone_mesh = self.scene.add_mesh_simple(
                f"{self.prefix}/backbone", vertices=vertices, faces=faces,
                color=self.rod_color, opacity=self.rod_opacity,
                flat_shading=False, side="front", material="toon5")
            self.backbone_mesh.cast_shadow = True
            self.backbone_mesh.receive_shadow = True
        else:
            self.backbone_mesh.vertices = vertices

    def update_backbone_ellipsoids(self, solution):
        if not self.plot_backbone_ellipsoids:
            return

        states = solution.states[::self.skip_backbone_ellipsoids]

        while len(self.backbone_ellipsoid_handles) < len(states):
            idx = len(self.backbone_ellipsoid_handles)
            handle = utils.add_ellipsoid(
                self.scene, f"{self.prefix}/backbone_ellipsoids/{idx}",
                position=(0, 0, 0), cov=np.eye(3) * 1e-8,
                color=utils.DEEP_CADMIUM_RED, opacity=0.2)
            self.backbone_ellipsoid_handles.append(handle)

        for handle, state in zip(self.backbone_ellipsoid_handles, states):
            pose, cov = state.pose.mean, state.pose.cov
            R, p = pose[:3, :3], pose[:3, 3]
            world_cov = R @ (cov[3:, 3:] @ R.T)
            utils.update_ellipsoid(handle, p, world_cov)

    def update_backbone_frames(self, solution):
        if not self.plot_backbone_frames:
            return

        for i, state in enumerate(solution.states):
            pose = state.pose.mean
            self.scene.add_frame(
                f"{self.prefix}/backbone_frames/{i}",
                position=pose[:3, 3], wxyz=utils.pose_to_wxyz(pose),
                axes_length=self.cartesian_frame_scale, axes_radius=self.cartesian_frame_scale * 0.08)

    def update_wrenches(self, solution):
        if not self.plot_wrenches:
            return

        states = solution.states if self.plot_base_wrench else solution.states[1:]
        states = [s for s in states if s.wrench is not None]

        while len(self.wrench_moment_ellipsoid_handles) < len(states):
            idx = len(self.wrench_moment_ellipsoid_handles)
            self.wrench_moment_ellipsoid_handles.append(utils.add_ellipsoid(
                self.scene, f"{self.prefix}/wrenches/{idx}/moment_ellipsoid",
                position=(0, 0, 0), cov=np.eye(3) * 1e-8, color=utils.CADMIUM_LEMON, opacity=0.4))
            self.wrench_force_ellipsoid_handles.append(utils.add_ellipsoid(
                self.scene, f"{self.prefix}/wrenches/{idx}/force_ellipsoid",
                position=(0, 0, 0), cov=np.eye(3) * 1e-8, color=utils.CADMIUM_LEMON, opacity=0.4))

        for i, state in enumerate(states):
            p = state.pose.mean[:3, 3]
            moment_mean, force_mean = state.wrench.mean[:3], state.wrench.mean[3:]
            moment_cov, force_cov = state.wrench.cov[:3, :3], state.wrench.cov[3:, 3:]

            utils.set_vector_arrow(
                self.scene, f"{self.prefix}/wrenches/{i}/moment_arrow",
                p, moment_mean, self.moment_scale, color=utils.DEEP_PINK,
                shaft_radius=self.arrow_shaft_radius)
            utils.set_vector_arrow(
                self.scene, f"{self.prefix}/wrenches/{i}/force_arrow",
                p, force_mean, self.force_scale, color=utils.DARK_ORCHID,
                shaft_radius=self.arrow_shaft_radius)

            utils.update_ellipsoid(
                self.wrench_moment_ellipsoid_handles[i],
                p + moment_mean * self.moment_scale, moment_cov, scale=self.force_scale)
            utils.update_ellipsoid(
                self.wrench_force_ellipsoid_handles[i],
                p + force_mean * self.force_scale, force_cov, scale=self.force_scale)

    def update(self, solution):
        self.update_base_plate(solution)
        self.update_tip_plate(solution)
        self.update_rod_tube(solution)
        self.update_backbone_ellipsoids(solution)
        self.update_wrenches(solution)
        self.update_backbone_frames(solution)


class ViserCosseratRodPlotter:
    def __init__(self,
                 server=None,
                 port=8080,
                 plot_base_plate=True,
                 plot_tip_plate=False,
                 plot_wrenches=True,
                 plot_base_wrench=False,
                 plot_backbone_frames=False,
                 plot_backbone_ellipsoids=True,
                 backbone_radius=0.005,
                 moment_scale=0.2,
                 force_scale=0.1,
                 base_plate_size=0.1,
                 cartesian_frame_scale=0.03):

        # Owns its own server if one isn't given, the same way a pyvista
        # plotter owns its own window -- lets standalone scripts construct a
        # plotter directly, with no separate viser import/setup of their own.
        if server is None:
            server = viser.ViserServer(port=port)
            print("Open the URL above in a browser to watch.")
        self.server = server
        utils.setup_default_lighting(server)

        self.mesh_manager = ViserCosseratRodMeshManager(
            server.scene,
            plot_base_plate=plot_base_plate,
            plot_tip_plate=plot_tip_plate,
            plot_wrenches=plot_wrenches,
            plot_base_wrench=plot_base_wrench,
            plot_backbone_frames=plot_backbone_frames,
            plot_backbone_ellipsoids=plot_backbone_ellipsoids,
            backbone_radius=backbone_radius,
            moment_scale=moment_scale,
            force_scale=force_scale,
            base_plate_size=base_plate_size,
            cartesian_frame_scale=cartesian_frame_scale,
        )

    def update(self, solution):
        self.mesh_manager.update(solution.marginals)

    def close(self):
        pass
