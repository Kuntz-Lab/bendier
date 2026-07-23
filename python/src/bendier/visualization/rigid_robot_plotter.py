# TODO: review this file
import numpy as np

from . import utils
from .base_plotter import BasePlotter


class RigidRobotMeshManager:
    def __init__(self,
                 scene,
                 prefix="/rigid_robot",
                 plot_link_ellipsoids=True,
                 plot_tip_wrench=True,
                 plot_joint_torques=True,
                 moment_scale=0.2,
                 force_scale=0.1,
                 torque_scale=0.3,
                 arrow_shaft_radius=0.004):

        self.scene = scene
        self.prefix = prefix

        self.plot_link_ellipsoids = plot_link_ellipsoids
        self.plot_tip_wrench = plot_tip_wrench
        self.plot_joint_torques = plot_joint_torques

        self.moment_scale = moment_scale
        self.force_scale = force_scale
        self.torque_scale = torque_scale
        self.arrow_shaft_radius = arrow_shaft_radius

        self.link_ellipsoid_batch = None
        self.tip_moment_arrow = None
        self.tip_force_arrow = None
        self.tip_moment_ellipsoid = None
        self.tip_force_ellipsoid = None
        self.joint_torque_arrows_batch = None
        self.joint_torque_ellipsoid_batch = None

    def update_link_ellipsoids(self, links, tip_pose):
        if not self.plot_link_ellipsoids:
            return

        # Actuated links plus the true tip pose (a separate variable, past
        # the last actuated link's fixed-but-uncertain tip offset) -- so the
        # tip's own (generally larger) uncertainty is visible too.
        poses = [link.pose for link in links] + [tip_pose]
        positions = np.array([pose.mean[:3, 3] for pose in poses])
        world_covs = np.array([
            pose.mean[:3, :3] @ pose.cov[3:, 3:] @ pose.mean[:3, :3].T
            for pose in poses
        ])

        if self.link_ellipsoid_batch is None:
            self.link_ellipsoid_batch = utils.add_ellipsoid_batch(
                self.scene, f"{self.prefix}/link_ellipsoids",
                positions, world_covs, color=utils.DEEP_CADMIUM_RED, opacity=0.25)
        else:
            utils.update_ellipsoid_batch(self.link_ellipsoid_batch, positions, world_covs)

    def update_tip_wrench(self, tip_pose_mean, tip_wrench):
        if not self.plot_tip_wrench or tip_wrench is None:
            return

        position = tip_pose_mean[:3, 3]
        moment_mean = tip_wrench.mean[:3]
        force_mean = tip_wrench.mean[3:]
        moment_cov = tip_wrench.cov[:3, :3]
        force_cov = tip_wrench.cov[3:, 3:]

        self.tip_moment_arrow = utils.set_vector_arrow(
            self.scene, f"{self.prefix}/tip_wrench/moment_arrow", position, moment_mean,
            self.moment_scale, color=utils.DEEP_PINK, shaft_radius=self.arrow_shaft_radius,
            handle=self.tip_moment_arrow)
        self.tip_force_arrow = utils.set_vector_arrow(
            self.scene, f"{self.prefix}/tip_wrench/force_arrow", position, force_mean,
            self.force_scale, color=utils.DARK_ORCHID, shaft_radius=self.arrow_shaft_radius,
            handle=self.tip_force_arrow)

        moment_ellipsoid_position = position + moment_mean * self.moment_scale
        force_ellipsoid_position = position + force_mean * self.force_scale

        if self.tip_moment_ellipsoid is None:
            self.tip_moment_ellipsoid = utils.add_ellipsoid(
                self.scene, f"{self.prefix}/tip_wrench/moment_ellipsoid",
                moment_ellipsoid_position, moment_cov, color=utils.CADMIUM_LEMON,
                opacity=0.4, scale=self.moment_scale)
            self.tip_force_ellipsoid = utils.add_ellipsoid(
                self.scene, f"{self.prefix}/tip_wrench/force_ellipsoid",
                force_ellipsoid_position, force_cov, color=utils.CADMIUM_LEMON,
                opacity=0.4, scale=self.force_scale)
        else:
            utils.update_ellipsoid(
                self.tip_moment_ellipsoid, moment_ellipsoid_position, moment_cov, scale=self.moment_scale)
            utils.update_ellipsoid(
                self.tip_force_ellipsoid, force_ellipsoid_position, force_cov, scale=self.force_scale)

    def update_joint_torques(self, links, joint_axes, joint_torques):
        if not self.plot_joint_torques or joint_torques is None:
            return

        # Joint i's torque acts at link i+1 (its child link), along that
        # link's own world-frame axis direction -- pose_child's orientation
        # always agrees with the joint's own frame on axis direction (see
        # RigidJointTorqueFactor's docstring), so no separate offset lookup
        # is needed here either.
        positions = np.array([links[i + 1].pose.mean[:3, 3] for i in range(len(joint_axes))])
        axes_world = np.array([
            links[i + 1].pose.mean[:3, :3] @ joint_axes[i] for i in range(len(joint_axes))
        ])
        torque_vecs = axes_world * joint_torques.mean[:, None]
        torque_vars = np.diag(joint_torques.cov)

        # A joint-torque sensor only ever constrains its own axis direction
        # -- drawn as a needle-shaped (rank-1 covariance) ellipsoid rather
        # than a proper 3D one, since there's genuinely no information
        # along the other two directions.
        world_covs = torque_vars[:, None, None] * (axes_world[:, :, None] @ axes_world[:, None, :])

        self.joint_torque_arrows_batch = utils.set_vector_arrows_batch(
            self.scene, f"{self.prefix}/joint_torques/arrows",
            positions, torque_vecs, self.torque_scale, color=utils.DEEP_PINK,
            shaft_radius=self.arrow_shaft_radius, handle=self.joint_torque_arrows_batch)

        ellipsoid_positions = positions + torque_vecs * self.torque_scale

        if self.joint_torque_ellipsoid_batch is None:
            self.joint_torque_ellipsoid_batch = utils.add_ellipsoid_batch(
                self.scene, f"{self.prefix}/joint_torques/ellipsoids",
                ellipsoid_positions, world_covs, color=utils.CADMIUM_LEMON,
                opacity=0.4, scale=self.torque_scale)
        else:
            utils.update_ellipsoid_batch(
                self.joint_torque_ellipsoid_batch, ellipsoid_positions, world_covs, scale=self.torque_scale)

    def update(self, marginals, joint_axes=None):
        self.update_link_ellipsoids(marginals.links, marginals.tip_pose)
        self.update_tip_wrench(marginals.tip_pose.mean, marginals.tip_wrench)
        if joint_axes is not None:
            self.update_joint_torques(marginals.links, joint_axes, marginals.joint_torques)


class RigidRobotPlotter(BasePlotter):
    def __init__(self,
                 server=None,
                 port=8080,
                 joint_axes=None,
                 prefix="/rigid_robot",
                 plot_link_ellipsoids=True,
                 plot_tip_wrench=True,
                 plot_joint_torques=True,
                 moment_scale=0.2,
                 force_scale=0.1,
                 torque_scale=0.3,
                 show_solve_stats=True):
        super().__init__(server=server, port=port, show_solve_stats=show_solve_stats)

        self.joint_axes = joint_axes

        self.mesh_manager = RigidRobotMeshManager(
            self.server.scene,
            prefix=prefix,
            plot_link_ellipsoids=plot_link_ellipsoids,
            plot_tip_wrench=plot_tip_wrench,
            plot_joint_torques=plot_joint_torques,
            moment_scale=moment_scale,
            force_scale=force_scale,
            torque_scale=torque_scale,
        )

    def update(self, solution):
        self.mesh_manager.update(solution.marginals, self.joint_axes)
        self.update_solve_stats(solution.meta)
