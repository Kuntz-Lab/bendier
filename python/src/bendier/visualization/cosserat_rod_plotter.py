import numpy as np
import viser

from . import utils

BACKBONE_SIDES = 16
END_PLATE_SIDES = 64


class CosseratRodMeshManager:
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
                 arrow_shaft_radius=None,
                 tube_segments_per_interval=6):

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
        # How many interpolated rings to insert between each pair of actual
        # solver nodes -- see utils.interpolate_poses. 1 disables it (one
        # ring per node, the old behavior).
        self.tube_segments_per_interval = tube_segments_per_interval

        self.backbone_mesh = None
        self.base_plate_mesh = None
        self.tip_plate_mesh = None
        self.backbone_ellipsoid_batch = None
        self.backbone_frames_batch = None
        self.wrench_moment_ellipsoid_batch = None
        self.wrench_force_ellipsoid_batch = None
        self.wrench_moment_arrows_batch = None
        self.wrench_force_arrows_batch = None

    def update_base_plate(self, solution):
        if not self.plot_base_plate:
            return

        pose = solution.states[0].pose.mean
        half_thick = (self.base_plate_size / 10.0) / 2.0
        if self.base_plate_mesh is None:
            self.base_plate_mesh = utils.add_disc(
                self.scene, f"{self.prefix}/base_plate", pose,
                self.base_plate_size / 2.0, half_thick, utils.SILVER,
                radial_segments=END_PLATE_SIDES)
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
                radial_segments=END_PLATE_SIDES)
        else:
            utils.update_disc(self.tip_plate_mesh, pose)

    def update_rod_tube(self, solution):
        poses = [state.pose.mean for state in solution.states]
        dense_poses = utils.interpolate_poses(poses, self.tube_segments_per_interval)
        vertices = utils.tube_vertices(dense_poses, self.backbone_radius, BACKBONE_SIDES)

        if self.backbone_mesh is None:
            faces = utils.tube_faces(len(dense_poses), BACKBONE_SIDES)
            self.backbone_mesh = self.scene.add_mesh_simple(
                f"{self.prefix}/backbone", vertices=vertices, faces=faces,
                color=self.rod_color, opacity=self.rod_opacity,
                flat_shading=False, side="front", material="toon5")
        else:
            self.backbone_mesh.vertices = vertices

    def update_backbone_ellipsoids(self, solution):
        if not self.plot_backbone_ellipsoids:
            return

        states = solution.states[::self.skip_backbone_ellipsoids]
        positions = np.array([s.pose.mean[:3, 3] for s in states])
        world_covs = np.array([
            s.pose.mean[:3, :3] @ s.pose.cov[3:, 3:] @ s.pose.mean[:3, :3].T
            for s in states
        ])

        if self.backbone_ellipsoid_batch is None:
            self.backbone_ellipsoid_batch = utils.add_ellipsoid_batch(
                self.scene, f"{self.prefix}/backbone_ellipsoids",
                positions, world_covs, color=utils.DEEP_CADMIUM_RED, opacity=0.2)
        else:
            utils.update_ellipsoid_batch(self.backbone_ellipsoid_batch, positions, world_covs)

    def update_backbone_frames(self, solution):
        if not self.plot_backbone_frames:
            return

        # One batched scene node for every node's frame, rather than one
        # add_frame() per node -- viser's own docs call add_frame-in-a-loop
        # out as the slow path (see add_batched_axes' docstring).
        poses = np.array([s.pose.mean for s in solution.states])
        positions = poses[:, :3, 3]
        wxyzs = utils.pose_batch_to_wxyz(poses)

        if self.backbone_frames_batch is None:
            self.backbone_frames_batch = self.scene.add_batched_axes(
                f"{self.prefix}/backbone_frames", batched_wxyzs=wxyzs, batched_positions=positions,
                axes_length=self.cartesian_frame_scale, axes_radius=self.cartesian_frame_scale * 0.08)
        else:
            self.backbone_frames_batch.batched_positions = positions
            self.backbone_frames_batch.batched_wxyzs = wxyzs

    def update_wrenches(self, solution):
        if not self.plot_wrenches:
            return

        states = solution.states if self.plot_base_wrench else solution.states[1:]
        states = [s for s in states if s.wrench is not None]
        if not states:
            return

        positions = np.array([s.pose.mean[:3, 3] for s in states])
        moment_means = np.array([s.wrench.mean[:3] for s in states])
        force_means = np.array([s.wrench.mean[3:] for s in states])
        moment_covs = np.array([s.wrench.cov[:3, :3] for s in states])
        force_covs = np.array([s.wrench.cov[3:, 3:] for s in states])

        self.wrench_moment_arrows_batch = utils.set_vector_arrows_batch(
            self.scene, f"{self.prefix}/wrenches/moment_arrows",
            positions, moment_means, self.moment_scale, color=utils.DEEP_PINK,
            shaft_radius=self.arrow_shaft_radius, handle=self.wrench_moment_arrows_batch)
        self.wrench_force_arrows_batch = utils.set_vector_arrows_batch(
            self.scene, f"{self.prefix}/wrenches/force_arrows",
            positions, force_means, self.force_scale, color=utils.DARK_ORCHID,
            shaft_radius=self.arrow_shaft_radius, handle=self.wrench_force_arrows_batch)

        moment_ellipsoid_positions = positions + moment_means * self.moment_scale
        force_ellipsoid_positions = positions + force_means * self.force_scale

        if self.wrench_moment_ellipsoid_batch is None:
            self.wrench_moment_ellipsoid_batch = utils.add_ellipsoid_batch(
                self.scene, f"{self.prefix}/wrenches/moment_ellipsoids",
                moment_ellipsoid_positions, moment_covs,
                color=utils.CADMIUM_LEMON, opacity=0.4, scale=self.force_scale)
            self.wrench_force_ellipsoid_batch = utils.add_ellipsoid_batch(
                self.scene, f"{self.prefix}/wrenches/force_ellipsoids",
                force_ellipsoid_positions, force_covs,
                color=utils.CADMIUM_LEMON, opacity=0.4, scale=self.force_scale)
        else:
            utils.update_ellipsoid_batch(
                self.wrench_moment_ellipsoid_batch, moment_ellipsoid_positions, moment_covs,
                scale=self.force_scale)
            utils.update_ellipsoid_batch(
                self.wrench_force_ellipsoid_batch, force_ellipsoid_positions, force_covs,
                scale=self.force_scale)

    def update(self, solution):
        self.update_base_plate(solution)
        self.update_tip_plate(solution)
        self.update_rod_tube(solution)
        self.update_backbone_ellipsoids(solution)
        self.update_wrenches(solution)
        self.update_backbone_frames(solution)


class CosseratRodPlotter:
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

        self.mesh_manager = CosseratRodMeshManager(
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
