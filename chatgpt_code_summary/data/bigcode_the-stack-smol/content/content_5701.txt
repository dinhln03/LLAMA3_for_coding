import numpy as np
import os
from sklearn.neighbors import NearestNeighbors

from pydrake.multibody.rigid_body import RigidBody
from pydrake.all import (
        AddFlatTerrainToWorld,
        AddModelInstancesFromSdfString,
        AddModelInstanceFromUrdfFile,
        FindResourceOrThrow,
        FloatingBaseType,
        InputPort,
        Isometry3,
        OutputPort,
        RgbdCamera,
        RigidBodyPlant,
        RigidBodyTree,
        RigidBodyFrame,
        RollPitchYaw,
        RollPitchYawFloatingJoint,
        RotationMatrix,
        Value,
        VisualElement,
    )

import meshcat
import meshcat.transformations as tf
import meshcat.geometry as g


# From
# https://www.opengl.org/discussion_boards/showthread.php/197893-View-and-Perspective-matrices
def normalize(x):
    return x / np.linalg.norm(x)


def save_pointcloud(pc, normals, path):
    joined = np.hstack([pc.T, normals.T])
    np.savetxt(path, joined)


def load_pointcloud(path):
    joined = np.loadtxt(path)
    return joined[:, 0:3].T, joined[:, 3:6].T


def translate(x):
    T = np.eye(4)
    T[0:3, 3] = x[:3]
    return T


def get_pose_error(tf_1, tf_2):
    rel_tf = transform_inverse(tf_1).dot(tf_2)
    if np.allclose(np.diag(rel_tf[0:3, 0:3]), [1., 1., 1.]):
        angle_dist = 0.
    else:
        # Angle from rotation matrix
        angle_dist = np.arccos(
            (np.sum(np.diag(rel_tf[0:3, 0:3])) - 1) / 2.)
    euclid_dist = np.linalg.norm(rel_tf[0:3, 3])
    return euclid_dist, angle_dist


# If misalignment_tol = None, returns the average
# distance between the model clouds when transformed
# by est_tf and gt_tf (using nearest-point lookups
# for each point in the gt-tf'd model cloud).
# If misalignment_tol is a number, it returns
# the percent of points that are misaligned by more
# than the misalignment error under the same distance
# metric.
def get_earth_movers_error(est_tf, gt_tf, model_cloud,
                           misalignment_tol=0.005):
    # Transform the model cloud into both frames
    est_model_cloud = transform_points(est_tf, model_cloud)
    gt_model_cloud = transform_points(gt_tf, model_cloud)
    # For every point in the model cloud, find the distance
    # to the closest point in the estimated model cloud,
    # as a way of finding the swept volume between the
    # models in those poses.
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(gt_model_cloud.T)
    dist, _ = neigh.kneighbors(
        est_model_cloud[0:3, :].T, return_distance=True)
    if misalignment_tol is None:
        return np.mean(dist)
    else:
        return np.mean(dist > misalignment_tol)


def draw_points(vis, vis_prefix, name, points,
                normals=None, colors=None, size=0.001,
                normals_length=0.01):
    vis[vis_prefix][name].set_object(
        g.PointCloud(position=points,
                     color=colors,
                     size=size))
    n_pts = points.shape[1]
    if normals is not None:
        # Drawing normals for debug
        lines = np.zeros([3, n_pts*2])
        inds = np.array(range(0, n_pts*2, 2))
        lines[:, inds] = points[0:3, :]
        lines[:, inds+1] = points[0:3, :] + \
            normals * normals_length
        vis[vis_prefix]["%s_normals" % name].set_object(
            meshcat.geometry.LineSegmentsGeometry(
                lines, None))


def transform_points(tf, pts):
    return ((tf[:3, :3].dot(pts).T) + tf[:3, 3]).T


def transform_inverse(tf):
    new_tf = np.eye(4)
    new_tf[:3, :3] = tf[:3, :3].T
    new_tf[:3, 3] = -new_tf[:3, :3].dot(tf[:3, 3])
    return new_tf


def lookat(eye, target, up):
    # For a camera with +x right, +y down, and +z forward.
    eye = np.array(eye)
    target = np.array(target)
    up = np.array(up)
    F = target[:3] - eye[:3]
    f = normalize(F)
    U = normalize(up[:3])
    s = np.cross(f, U)  # right
    u = np.cross(s, f)  # up
    M = np.eye(4)
    M[:3, :3] = np.vstack([s, -u, f]).T

    # OLD:
    # flip z -> x
    # -x -> y
    # -y -> z
    # CAMERA FORWARD is +x-axis
    # CAMERA RIGHT is -y axis
    # CAMERA UP is +z axis
    # Why does the Drake documentation lie to me???
    T = translate(eye)
    return T.dot(M)


def add_single_instance_to_rbt(
        rbt, config, instance_config, i,
        floating_base_type=FloatingBaseType.kRollPitchYaw):
    class_name = instance_config["class"]
    if class_name not in config["objects"].keys():
        raise ValueError("Class %s not in classes." % class_name)
    if len(instance_config["pose"]) != 6:
        raise ValueError("Class %s has pose size != 6. Use RPY plz" %
                         class_name)
    frame = RigidBodyFrame(
        "%s_%d" % (class_name, i), rbt.world(),
        instance_config["pose"][0:3],
        instance_config["pose"][3:6])
    model_path = config["objects"][class_name]["model_path"]
    _, extension = os.path.splitext(model_path)
    if extension == ".urdf":
        AddModelInstanceFromUrdfFile(
            model_path, floating_base_type, frame, rbt)
    elif extension == ".sdf":
        AddModelInstancesFromSdfString(
            open(model_path).read(), floating_base_type, frame, rbt)
    else:
        raise ValueError("Class %s has non-sdf and non-urdf model name." %
                         class_name)


def setup_scene(rbt, config):
    if config["with_ground"] is True:
        AddFlatTerrainToWorld(rbt)

    for i, instance_config in enumerate(config["instances"]):
        add_single_instance_to_rbt(rbt, config, instance_config, i,
                                   floating_base_type=FloatingBaseType.kFixed)
    # Add camera geometry!
    camera_link = RigidBody()
    camera_link.set_name("camera_link")
    # necessary so this last link isn't pruned by the rbt.compile() call
    camera_link.set_spatial_inertia(np.eye(6))
    camera_link.add_joint(
        rbt.world(),
        RollPitchYawFloatingJoint(
            "camera_floating_base",
            np.eye(4)))
    rbt.add_rigid_body(camera_link)

    # - Add frame for camera fixture.
    camera_frame = RigidBodyFrame(
        name="rgbd_camera_frame", body=camera_link,
        xyz=[0.0, 0., 0.], rpy=[0., 0., 0.])
    rbt.addFrame(camera_frame)
    rbt.compile()