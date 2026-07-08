import itertools

import pyvista as pv
import vtk
import numpy as np

from . import utils
from .cosserat_rod_plotter import CosseratRodMeshManager

ROD_COLORS = ['rebeccapurple', 'deepcadmiumred', 'cadmiumorange', 'lightcadmiumyellow', 'seagreen', 'royalblue']


class ParallelRobotPlotter:
    def __init__(self,
                 platform_z_offset=0.0,
                 plot_rod_wrenches=True,
                 plot_tip_force=True,
                 plot_base_wrenches=False,
                 plot_backbone_frames=False,
                 plot_backbone_ellipsoids=True,
                 **kwargs):

        self.plotter = utils.PlotterBase(**kwargs)

        self.rod_managers = None

        self.moment_scale = 0.2
        self.force_scale = 0.3
        self.platform_z_offset = platform_z_offset
        self.plot_tip_force = plot_tip_force
        self._plot_rod_wrenches = plot_rod_wrenches
        self._plot_base_wrenches = plot_base_wrenches
        self._plot_backbone_frames = plot_backbone_frames
        self._plot_backbone_ellipsoids = plot_backbone_ellipsoids

        base_plate = pv.Cylinder(direction=(0,0,1), radius=0.2, height=0.02)
        self.plotter.plotter.add_mesh(base_plate, color="silver", show_edges=True, line_width=2, opacity=0.3)

    def _ensure_rod_managers(self, num_rods):
        if self.rod_managers is not None:
            return

        self.rod_managers = []
        colors = itertools.cycle(ROD_COLORS)
        for _ in range(num_rods):
            self.rod_managers.append(CosseratRodMeshManager(
                plot_base_plate=False,
                plot_wrenches=self._plot_rod_wrenches,
                plot_base_wrench=self._plot_base_wrenches,
                plot_backbone_frames=self._plot_backbone_frames,
                plot_backbone_ellipsoids=self._plot_backbone_ellipsoids,
                backbone_radius=0.005,
                moment_scale=self.moment_scale,
                force_scale=self.force_scale,
                cartesian_frame_scale=0.025,
                rod_opacity=0.5,
                rod_color=next(colors)
                )
            )

    def update_platform(self, solution, plotter):

        if plotter.frame == 0:
            mesh = pv.Cylinder(direction=(0,0,1), radius=0.15, height=0.01)
            mesh.points = mesh.points + np.array([0, 0, self.platform_z_offset])
            actor = plotter.plotter.add_mesh(mesh, color="silver", show_edges=True, line_width=2, opacity=0.3)
            self.platform_transform = vtk.vtkTransform()
            actor.SetUserTransform(self.platform_transform)

            mesh = pv.Cylinder(direction=(0,0,1), radius=0.005, height=np.abs(self.platform_z_offset))
            mesh.points = mesh.points + np.array([0, 0, self.platform_z_offset / 2])
            actor = plotter.plotter.add_mesh(mesh, color="silver")
            actor.SetUserTransform(self.platform_transform)

            mesh = pv.Sphere(radius=0.01)
            actor = plotter.plotter.add_mesh(mesh, color="silver")
            actor.SetUserTransform(self.platform_transform)

            # axes = utils.get_axes_frame(length=0.1)
            # for arrow, color in zip(axes, utils.frame_arrow_colors):
            #     actor = plotter.plotter.add_mesh(arrow, color=color)
            #     actor.SetUserTransform(self.platform_transform)
            
            mesh = pv.Sphere(radius=1)
            actor = plotter.plotter.add_mesh(mesh, color="deepcadmiumred", lighting=False, opacity=0.2)
            self.platform_ellipsoid_transform = vtk.vtkTransform()
            actor.SetUserTransform(self.platform_ellipsoid_transform)

        pose = solution.platform_pose.mean
        self.platform_transform.SetMatrix(pose.flatten().tolist())
        
        p = pose[:3,3]
        R = pose[:3,:3]
        cov = solution.platform_pose.cov
        cov = R @ (cov[3:, 3:] @ R.T)
        T = utils.get_ellipsoid_transform(p, cov)
        self.platform_ellipsoid_transform.SetMatrix(T.flatten().tolist())

    def update_tip_force(self, solution, plotter, tip_force_gt):
        if not self.plot_tip_force:
            return
        
        if plotter.frame == 0:
            shaft_scale = 0.08
            mesh = utils.get_arrow(shaft_scale=shaft_scale)
            actor = plotter.plotter.add_mesh(mesh, color='darkorchid', lighting=False)
            self.tip_force_arrow_transform = vtk.vtkTransform()
            actor.SetUserTransform(self.tip_force_arrow_transform)

            mesh = mesh = pv.Sphere(radius=1)
            actor = plotter.plotter.add_mesh(mesh, color="cadmiumlemon", lighting=False, opacity=0.4)
            self.tip_force_ellipsoid_transform = vtk.vtkTransform()
            actor.SetUserTransform(self.tip_force_ellipsoid_transform)

            if tip_force_gt is not None:
                mesh = utils.get_arrow(shaft_scale=shaft_scale)
                self.tip_force_gt_transform = vtk.vtkTransform()
                actor = self.plotter.plotter.add_mesh(mesh, color='green', lighting=False)
                actor.SetUserTransform(self.tip_force_gt_transform)
            
        # Update vtkTransforms for each actor
        p = solution.platform_pose.mean[:3,3]
        f = solution.platform_wrench.mean[3:]
        cov = solution.platform_wrench.cov[3:, 3:]

        matrix = utils.get_arrow_transform(p, f, scale=self.force_scale)
        self.tip_force_arrow_transform.SetMatrix(matrix.flatten().tolist())

        matrix = utils.get_ellipsoid_transform(p + f * self.force_scale, cov, scale=self.force_scale)
        self.tip_force_ellipsoid_transform.SetMatrix(matrix.flatten().tolist())

        if tip_force_gt is not None:
            matrix = utils.get_arrow_transform(p, tip_force_gt, scale=self.force_scale)
            self.tip_force_gt_transform.SetMatrix(matrix.flatten().tolist())

    def update(self, solution, tip_force_gt=None):
        self._ensure_rod_managers(len(solution.marginals.rods))
        for i, manager in enumerate(self.rod_managers):
            manager.update(solution.marginals.rods[i], self.plotter)

        self.update_platform(solution.marginals, self.plotter)
        self.update_tip_force(solution.marginals, self.plotter, tip_force_gt)

        self.plotter.update(solution)

    def close(self):
        self.plotter.close()

