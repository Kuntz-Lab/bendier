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
            camera_focal_point=[0,0.1,0],
            camera_azimuth=15,
            camera_distance=0.6,
            **kwargs
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
            moment_scale=0.2
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

                # Exclude base "disc"
                if i > 0:
                    mesh = pv.Cylinder(direction=(0,0,1), radius=disc_radius, height=disc_width, resolution=8)
                    actor = self.plotter.plotter.add_mesh(mesh, color='cornflowerblue', opacity=0.2, show_edges=True, line_width=3.0)
                    actor.SetUserTransform(disc_transform)
                
                # Add holes to disc using same transform
                for angle in np.linspace(0, 2 * np.pi, num_holes_per_disc, endpoint=False):
                    hole_location = np.array([routing_radius * np.cos(angle), routing_radius * np.sin(angle), 0.0])
                    mesh = pv.Sphere(radius=hole_radius, center=hole_location)
                    actor = self.plotter.plotter.add_mesh(mesh, color='black', opacity=0.5, lighting=False)
                    actor.SetUserTransform(disc_transform)
        
        # Update vtkTransforms for each actor
        for ii in range(num_discs):
            T = solution.marginals.rod.states[disc_pose_idx[ii]].pose.mean
            self.disc_transforms[ii].SetMatrix(T.flatten().tolist())

    def update_tip_force(self, solution, tip_force_gt=None):
        if not self.plot_tip_force:
            return

        if self.plotter.frame == 0:
            shaft_scale=0.02

            mesh = utils.get_arrow(shaft_scale=shaft_scale)
            self.tip_force_arrow_transform = vtk.vtkTransform()
            actor = self.plotter.plotter.add_mesh(mesh, color='darkorchid', lighting=False)
            actor.SetUserTransform(self.tip_force_arrow_transform)

            mesh = pv.Sphere(radius=1)
            self.tip_force_ellipsoid_transform = vtk.vtkTransform()
            actor = self.plotter.plotter.add_mesh(mesh, color="cadmiumlemon", lighting=False, opacity=0.4)
            actor.SetUserTransform(self.tip_force_ellipsoid_transform)

            if tip_force_gt is not None:
                mesh = utils.get_arrow(shaft_scale=shaft_scale)
                self.tip_force_gt_transform = vtk.vtkTransform()
                actor = self.plotter.plotter.add_mesh(mesh, color='green', lighting=False)
                actor.SetUserTransform(self.tip_force_gt_transform)


        # Update vtkTransforms for each actor
        p = solution.marginals.rod.states[-1].pose.mean[:3,3]
        f = solution.marginals.external_wrenches[-1].mean[3:]
        cov = solution.marginals.external_wrenches[-1].cov[3:,3:]

        force_scale = 0.3
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
            actor = self.plotter.plotter.add_mesh(mesh, color='red', lighting=False)
            actor.SetUserTransform(self.p_desired_transform)

        matrix = np.eye(4)
        matrix[:3,3] = p
        self.p_desired_transform.SetMatrix(matrix.flatten().tolist())

    def update(self, solution, p_desired=None, tip_force_gt=None):
        self.rod_manager.update(solution.marginals.rod, self.plotter)
        self.update_tendons(solution)
        self.update_discs(solution)
        self.update_tip_force(solution, tip_force_gt)
        
        if p_desired is not None:
            self.update_p_desired(p_desired)
        
        self.plotter.update(solution)




# def get_largest_norm(f_samples, f_gt, f_mean):

#     norms = []

#     if f_samples is not None:
#         norms.append(np.max(np.linalg.norm(f_samples, axis=1)))

#     for f in [f_gt, f_mean]:
#         if f is not None:
#             norms.append(np.linalg.norm(f))

#     max_norm = max(norms) if norms else 1.0  # Fallback value to avoid div-by-zero

#     return max_norm
    

# class TendonRobotPlotter:
#     def __init__(self, 
#                  title, 
#                  save_frames_mode=False,
#                  single_plot_mode=False,
#                  plot_tip_force=False, 
#                  plot_dist_load=False,
#                  plot_backbone_ellipsoids=True,
#                  waypoints=None, 
#                  cylinders=None, 
#                  azimuth=15,
#                  camera_distance=0.6,
#                  focal_point_y=0.12):
        
#         self.save_frames_mode = save_frames_mode
#         self.single_plot_mode = single_plot_mode
#         self.plot_tip_force = plot_tip_force
#         self.plot_dist_load = plot_dist_load
#         self.plot_backbone_ellipsoids = plot_backbone_ellipsoids

#         self.cylinders = cylinders
#         self.waypoints = waypoints

#         self.azimuth = azimuth
#         self.camera_distance = camera_distance
#         self.focal_point_y = focal_point_y

#         if save_frames_mode:
#             dir_name = title.strip().lower().replace(" ", "_")
#             self.frames_path = Path("videos") / "frames" / dir_name
#             shutil.rmtree(self.frames_path, ignore_errors=True)
#             self.frames_path.mkdir(parents=True, exist_ok=True)

#         self.window_size = (2000, 2000)
#         self.plotter = pv.Plotter(window_size=self.window_size, off_screen=save_frames_mode)
#         self.frame = 0
#         self.solve_time_ms_history = []
            
#     def init_scene(self, solution):
#         plate = get_base_plate(solution)
#         self.plotter.add_mesh(plate, color="silver", show_edges=True, line_width=2)
        
#         if self.waypoints is not None:
#             for point in self.waypoints:
#                 mesh = pv.Sphere(0.0015, center=point)
#                 self.plotter.add_mesh(mesh, color="red")
        
#         if self.cylinders is not None:
#             for cylinder in self.cylinders:
#                 mesh = pv.Cylinder(cylinder['center'], cylinder['z'], cylinder['radius'], cylinder['length'])
#                 self.plotter.add_mesh(mesh, smooth_shading=True, color='cadmiumyellow')

#         focal_point = np.array([0.0, self.focal_point_y, 0])
#         elevation = 15

#         az = np.deg2rad(self.azimuth)
#         el = np.deg2rad(elevation)

#         x = focal_point[0] + self.camera_distance * np.cos(el) * np.cos(az)
#         y = focal_point[1] + self.camera_distance * np.cos(el) * np.sin(az)
#         z = focal_point[2] + self.camera_distance * np.sin(el)

#         self.plotter.camera.position = (x, y, z)
#         self.plotter.camera.focal_point = focal_point

#         self.plotter.add_light(pv.Light(position=(1.0, 0.7, 0.5), intensity=0.5, light_type='scene light'))
#         self.plotter.add_light(pv.Light(position=(0.7, -1.0, 0.5), intensity=0.2, light_type='scene light'))
#         self.plotter.add_light(pv.Light(position=(-1.0, -1.0, 0.5), intensity=0.2, light_type='scene light'))

#         # self.plotter.add_axes()
#         self.plotter.enable_depth_peeling(10)
#         self.plotter.enable_anti_aliasing()

#         if not self.save_frames_mode:
#             interactive_update = not self.single_plot_mode
#             # interactive_update=True
#             self.plotter.show(auto_close=False, interactive_update=interactive_update)
    
#     def update(self, solution, p_desired=None, tip_force_gt=None):

#         backbone_radius = 0.1 * solution.tendon_disc_config.routing_radius
#         backbone = get_tube_poses(solution.backbone_pose_mean, radius=backbone_radius)
#         tendons, discs, holes = get_tendon_disc_meshes(solution)
#         backbone_ellipsoids = get_backbone_ellipsoids(solution)

#         if self.plot_tip_force:
#             tip_force_mean_mesh, tip_force_2_sigma_mesh, tip_force_gt_mesh = get_tip_force_meshes(solution, tip_force_gt)
        
#         if self.plot_dist_load:
#             dist_load_meshes = get_dist_load_meshes(solution)



#         if self.frame == 0:
#             self.backbone_mesh = backbone
#             self.plotter.add_mesh(self.backbone_mesh, color='ultramarine', opacity = 0.7)

#             self.tendon_meshes = tendons
#             self.disc_meshes = discs
#             tendon_colors = ["crimson", "forestgreen", "royalblue", "mediumorchid", "goldenrod", "deeppink"]

#             for i, disc in enumerate(self.disc_meshes):
#                 if i == 0: continue
#                 disc.compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=True, inplace=True)
#                 self.plotter.add_mesh(disc, color='cornflowerblue', opacity=0.2, show_edges=True, line_width=3.0)
            
#             for j, tendon in enumerate(self.tendon_meshes):
#                 color = tendon_colors[j]
#                 self.plotter.add_mesh(tendon, color=color)

#             self.hole_meshes = holes
#             for i, hole in enumerate(self.hole_meshes):
#                 opacity = 0.5 if i < 8 else 0.2
#                 self.plotter.add_mesh(hole, color='black', opacity=opacity, lighting=False)
            
#             if self.plot_backbone_ellipsoids:
#                 self.backbone_2_sigma_meshes = backbone_ellipsoids
#                 for ellipsoid in self.backbone_2_sigma_meshes:
#                     self.plotter.add_mesh(ellipsoid, color="deepcadmiumred", lighting=False, opacity=0.2)

#             if self.plot_tip_force:
#                 self.tip_force_mean_mesh = tip_force_mean_mesh
#                 self.plotter.add_mesh(self.tip_force_mean_mesh, color='darkorchid', lighting=False)

#                 self.tip_force_2_sigma_mesh = tip_force_2_sigma_mesh
#                 self.plotter.add_mesh(self.tip_force_2_sigma_mesh, color="cadmiumlemon", lighting=False, opacity=0.4)

#                 if tip_force_gt is not None:
#                     self.tip_force_gt_mesh = tip_force_gt_mesh
#                     self.plotter.add_mesh(self.tip_force_gt_mesh, color='limegreen', lighting=False)
            
#             if self.plot_dist_load:
#                 self.dist_load_meshes = dist_load_meshes
#                 for mesh in self.dist_load_meshes:
#                     self.plotter.add_mesh(mesh, color="darkorchid", lighting=False)



#             self.init_scene(solution)
#         else:
#             self.backbone_mesh.shallow_copy(backbone)



#             if self.plot_dist_load:
#                 for (mesh_self, mesh) in zip(self.dist_load_meshes, dist_load_meshes):
#                     mesh_self.shallow_copy(mesh)
        