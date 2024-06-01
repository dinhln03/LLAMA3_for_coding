#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from math import cos, sin, pi
from tqdm import tqdm
import open3d as o3d

def render(pointcloud_file_path, estimate_normals_radius, estimate_normals_max_nn):
    pointcloud = o3d.io.read_point_cloud(pointcloud_file_path, print_progress=True)
    pointcloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=estimate_normals_radius,
            max_nn=estimate_normals_max_nn))
    o3d.visualization.draw_geometries([pointcloud])
    return True

class Renderer(object):
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.render_center = None
        self.euler_angle = [0, 0, 0]
        return

    def getRotationMatrixFromEulerAngle(self, euler_angle):
        R_x = np.array([
            [1, 0, 0],
            [0, cos(euler_angle[0]), -sin(euler_angle[0])],
            [0, sin(euler_angle[0]), cos(euler_angle[0])]
        ])

        R_y = np.array([
            [cos(euler_angle[1]), 0, sin(euler_angle[1])],
            [0, 1, 0],
            [-sin(euler_angle[1]), 0, cos(euler_angle[1])]
        ])

        R_z = np.array([
            [cos(euler_angle[2]), -sin(euler_angle[2]), 0],
            [sin(euler_angle[2]), cos(euler_angle[2]), 0],
            [0, 0, 1]
        ])

        rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))
        return rotation_matrix

    def getRotateDirection(self, direction_vector, euler_angle):
        np_direction_vector = np.array(direction_vector)
        direction_vector_norm = np.linalg.norm(np_direction_vector)
        if direction_vector_norm == 0:
            print("[ERROR][Renderer::getRotateDirection]")
            print("\t direction_vector_norm is 0!")
            return None

        np_unit_direction_vector = np_direction_vector / direction_vector_norm

        rotation_matrix = self.getRotationMatrixFromEulerAngle(euler_angle)

        rotate_direction = np.dot(rotation_matrix, np_unit_direction_vector)
        return rotate_direction.tolist()

    def rotateVis(self, delta_rotate_angle):
        self.euler_angle[0] = 0
        self.euler_angle[1] = -10 * pi / 180.0
        self.euler_angle[2] += delta_rotate_angle * pi / 180.0

        ctr = self.vis.get_view_control()

        front_direction = self.getRotateDirection(
            [1, 0, 0], self.euler_angle)
        ctr.set_front(front_direction)

        up_direction = self.getRotateDirection(
            [0, 0, 1], self.euler_angle)
        ctr.set_up(up_direction)

        ctr.set_lookat(self.render_center)
        #  ctr.set_zoom(0.5)
        return True

    def render(self, show_labels, scene_pointcloud_file_path=None):
        delta_rotate_angle = 0.5

        if scene_pointcloud_file_path is not None:
            print("start reading floor and wall...")
            self.splitLabeledPoints(scene_pointcloud_file_path)

        rendered_pointcloud = o3d.geometry.PointCloud()

        render_points = []
        render_colors = []
        print("start create rendered pointcloud...")
        for i in tqdm(range(len(self.pointcloud_list))):
            points = np.asarray(self.pointcloud_list[i].points).tolist()
            if len(points) == 0:
                continue
            for point in points:
                render_points.append(point)
                render_colors.append(self.d3_40_colors_rgb[i % len(self.d3_40_colors_rgb)] / 255.0)

        if scene_pointcloud_file_path is not None:
            print("start create rendered floor...")
            for wall_point in tqdm(self.labeled_point_cluster_list[0]):
                if abs(wall_point[2]) > 0.01:
                    continue
                render_points.append(wall_point)
                render_colors.append(np.array([132, 133, 135], dtype=np.uint8) / 255.0)

        rendered_pointcloud.points = o3d.utility.Vector3dVector(np.array(render_points))
        rendered_pointcloud.colors = o3d.utility.Vector3dVector(np.array(render_colors))

        rendered_pointcloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        self.render_center = rendered_pointcloud.get_axis_aligned_bounding_box().get_center()

        self.vis.create_window(window_name="Open3D RenderObject")
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array([1, 1, 1])
        render_option.point_size = 1

        self.vis.add_geometry(rendered_pointcloud)
        while True:
            self.rotateVis(delta_rotate_angle)
            #  self.vis.update_geometry()
            self.vis.poll_events()
            self.vis.update_renderer()

            if ord('q') == cv2.waitKey(1):
                break
        self.vis.destroy_window()
        return True

    def saveRender(self, output_video_file_path):
        fps = 30
        video_width = 1920
        video_height = 1080
        delta_rotate_angle = 0.5

        if scene_pointcloud_file_path is not None:
            print("start reading floor and wall...")
            self.splitLabeledPoints(scene_pointcloud_file_path)

        rendered_pointcloud = o3d.geometry.PointCloud()

        render_points = []
        render_colors = []
        print("start create rendered pointcloud...")
        for i in tqdm(range(len(self.pointcloud_list))):
            points = np.asarray(self.pointcloud_list[i].points).tolist()
            if len(points) == 0:
                continue
            for point in points:
                render_points.append(point)
                render_colors.append(self.d3_40_colors_rgb[i % len(self.d3_40_colors_rgb)] / 255.0)

        if scene_pointcloud_file_path is not None:
            print("start create rendered floor...")
            for wall_point in tqdm(self.labeled_point_cluster_list[0]):
                if abs(wall_point[2]) > 0.01:
                    continue
                render_points.append(wall_point)
                render_colors.append(np.array([132, 133, 135], dtype=np.uint8) / 255.0)

        rendered_pointcloud.points = o3d.utility.Vector3dVector(np.array(render_points))
        rendered_pointcloud.colors = o3d.utility.Vector3dVector(np.array(render_colors))

        self.render_center = rendered_pointcloud.get_axis_aligned_bounding_box().get_center()

        self.vis.create_window(window_name="Open3D RenderObject")
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array([1, 1, 1])
        render_option.point_size = 1

        self.vis.add_geometry(rendered_pointcloud)

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(output_video_file_path, fourcc, fps, (video_width, video_height))
        for i in range(int(360 / delta_rotate_angle)):
            self.rotateVis(0.5)
            #  self.vis.update_geometry()
            self.vis.poll_events()
            self.vis.update_renderer()

            open3d_image = np.asarray(self.vis.capture_screen_float_buffer()) * 255.0
            cv_image = cv2.cvtColor(open3d_image, cv2.COLOR_RGB2BGR).astype(np.uint8)

            out.write(cv_image)

        self.vis.destroy_window()
        out.release()
        return True

