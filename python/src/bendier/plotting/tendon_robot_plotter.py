import numpy as np
import pyvista as pv
import vtk

from . import utils
from .cosserat_rod_plotter import CosseratRodMeshManager


class TendonRobotPlotter:
    def __init__(self,
                 plot_rod_wrenches=False,
                 plot_tip_force=True,
                 plot_base_wrenches=False,
                 plot_backbone_frames=False,
                 plot_backbone_ellipsoids=True,
                 **kwargs):

        self.plotter = utils.PlotterBase(
            camera_focal_point=[0, 0.1, 0],
            camera_azimuth=15,
            camera_distance=0.6,
            **kwargs,
        )

        self.plot_tip_force = plot_tip_force

        self.rod_manager = CosseratRodMeshManager(
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

    def update_tendons(self, solution):
        num_tendons = solution.marginals.tendon_config.num_tendons
        num_discs = solution.marginals.tendon_config.num_discs

        if self.plotter.frame == 0:
            tendon_colors = ["crimson", "forestgreen", "royalblue", "mediumorchid", "goldenrod", "deeppink"]
            self.tendon_meshes = []
            for i in range(num_tendons):
                points = np.zeros((num_discs, 3))
                mesh = pv.lines_from_points(points)
                self.plotter.plotter.add_mesh(mesh, line_width=6, color=tendon_colors[i])
                self.tendon_meshes.append(mesh)

        for jj in range(num_tendons):
            points = []
            for ii in range(num_discs):
                disc_pose_idx = solution.marginals.tendon_config.disc_pose_idx[ii]
                hole = solution.marginals.tendon_config.hole_locations[ii][jj]
                T = solution.marginals.rod.states[disc_pose_idx].pose.mean
                p_world = T[:3, :3] @ hole + T[:3, 3]
                points.append(p_world)
            self.tendon_meshes[jj].points[:] = points

    def update_discs(self, solution):
        num_discs = solution.marginals.tendon_config.num_discs
        disc_pose_idx = solution.marginals.tendon_config.disc_pose_idx

        if self.plotter.frame == 0:
            routing_radius = solution.marginals.tendon_config.routing_radius
            disc_radius = 1.3 * routing_radius
            disc_width = 0.3 * routing_radius
            hole_radius = 0.05 * routing_radius
            num_holes_per_disc = 8

            self.disc_transforms = []

            for i in range(num_discs):
                disc_transform = vtk.vtkTransform()
                self.disc_transforms.append(disc_transform)

                if i > 0:
                    mesh = pv.Cylinder(direction=(0, 0, 1), radius=disc_radius, height=disc_width, resolution=8)
                    actor = self.plotter.plotter.add_mesh(mesh, color="cornflowerblue", opacity=0.2, show_edges=True, line_width=3.0)
                    actor.SetUserTransform(disc_transform)

                for angle in np.linspace(0, 2 * np.pi, num_holes_per_disc, endpoint=False):
                    hole_location = np.array([routing_radius * np.cos(angle), routing_radius * np.sin(angle), 0.0])
                    mesh = pv.Sphere(radius=hole_radius, center=hole_location)
                    actor = self.plotter.plotter.add_mesh(mesh, color="black", opacity=0.5, lighting=False)
                    actor.SetUserTransform(disc_transform)

        for ii in range(num_discs):
            T = solution.marginals.rod.states[disc_pose_idx[ii]].pose.mean
            self.disc_transforms[ii].SetMatrix(T.flatten().tolist())

    def update_tip_force(self, solution, tip_force_gt=None):
        if not self.plot_tip_force:
            return

        force_scale = 0.3

        if self.plotter.frame == 0:
            shaft_scale = 0.02
            mesh = utils.get_arrow(shaft_scale=shaft_scale)
            self.tip_force_arrow_transform = vtk.vtkTransform()
            actor = self.plotter.plotter.add_mesh(mesh, color="darkorchid", lighting=False)
            actor.SetUserTransform(self.tip_force_arrow_transform)

            mesh = pv.Sphere(radius=1)
            self.tip_force_ellipsoid_transform = vtk.vtkTransform()
            actor = self.plotter.plotter.add_mesh(mesh, color="cadmiumlemon", lighting=False, opacity=0.4)
            actor.SetUserTransform(self.tip_force_ellipsoid_transform)

            if tip_force_gt is not None:
                mesh = utils.get_arrow(shaft_scale=shaft_scale)
                self.tip_force_gt_transform = vtk.vtkTransform()
                actor = self.plotter.plotter.add_mesh(mesh, color="green", lighting=False)
                actor.SetUserTransform(self.tip_force_gt_transform)

        p = solution.marginals.rod.states[-1].pose.mean[:3, 3]
        f = solution.marginals.external_wrenches[-1].mean[3:]
        cov = solution.marginals.external_wrenches[-1].cov[3:, 3:]

        matrix = utils.get_arrow_transform(p, f, scale=force_scale)
        self.tip_force_arrow_transform.SetMatrix(matrix.flatten().tolist())

        matrix = utils.get_ellipsoid_transform(p + f * force_scale, cov, scale=force_scale)
        self.tip_force_ellipsoid_transform.SetMatrix(matrix.flatten().tolist())

        if tip_force_gt is not None:
            matrix = utils.get_arrow_transform(p, tip_force_gt, scale=force_scale)
            self.tip_force_gt_transform.SetMatrix(matrix.flatten().tolist())

    def update_p_desired(self, p):
        if self.plotter.frame == 0:
            mesh = pv.Sphere(radius=0.002)
            self.p_desired_transform = vtk.vtkTransform()
            actor = self.plotter.plotter.add_mesh(mesh, color="red", lighting=False)
            actor.SetUserTransform(self.p_desired_transform)

        matrix = np.eye(4)
        matrix[:3, 3] = p
        self.p_desired_transform.SetMatrix(matrix.flatten().tolist())

    def update(self, solution, p_desired=None, tip_force_gt=None):
        self.rod_manager.update(solution.marginals.rod, self.plotter)
        self.update_tendons(solution)
        self.update_discs(solution)
        self.update_tip_force(solution, tip_force_gt)

        if p_desired is not None:
            self.update_p_desired(p_desired)

        self.plotter.update(solution)

    def close(self):
        self.plotter.close()
