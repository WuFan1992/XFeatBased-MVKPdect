
from scipy.spatial import cKDTree
import cv2
import numpy as np
import os

from validation.hloc.utils.read_colmaps import *




def generate_infomap(images_bin):
    infomap = {}
    for image_id, image_info in images_bin.items():
        infomap[image_info.name] = {
            'qvec': image_info.qvec,
            'tvec': image_info.tvec
        }
    return infomap


def get_pose(ref_name, infomap):
    if ref_name in infomap:
        qvec = infomap[ref_name]['qvec']
        tvec = infomap[ref_name]['tvec']
        R = qvec2rotmat(qvec)
        t = np.array(tvec)
        
        return R, t
    raise ValueError(f"Reference image {ref_name} not found in COLMAP data.")


def load_colmap_model(model_path):

    images = read_extrinsics_binary(
        os.path.join(model_path, "images.bin")
    )

    xyzs, rgbs, errors, ids_3d, ids_img, ids_2dpts = \
        read_points3D_binary(
            os.path.join(model_path, "points3D.bin")
        )

    pointid_to_xyz = {}

    for pid, xyz in zip(ids_3d.flatten(), xyzs):
        pointid_to_xyz[int(pid)] = xyz

    name_to_image = {}

    for image in images.values():
        name_to_image[image.name] = image

    return images, name_to_image, pointid_to_xyz


def build_reference_kdtree(ref_image):

    xy = ref_image.xys

    tree = cKDTree(xy)

    return tree


def find_visible_3d_point(
        rpt,
        tree,
        ref_image,
        pointid_to_xyz,
        radius=8):

    idxs = tree.query_ball_point(rpt, r=radius)

    if len(idxs) == 0:
        return None

    best_xyz = None
    best_dist = 1e10

    for idx in idxs:

        pid = int(ref_image.point3D_ids[idx])

        if pid == -1:
            continue

        if pid not in pointid_to_xyz:
            continue

        colmap_xy = ref_image.xys[idx]

        dist = np.linalg.norm(colmap_xy - rpt)

        if dist < best_dist:
            best_dist = dist
            best_xyz = pointid_to_xyz[pid]

    return best_xyz


def build_2d3d_correspondence(
        kpts0,
        kpts1,
        ref_image,
        pointid_to_xyz):

    tree = build_reference_kdtree(ref_image)

    query_pts = []
    world_pts = []

    for qpt, rpt in zip(kpts0, kpts1):

        xyz = find_visible_3d_point(
            rpt,
            tree,
            ref_image,
            pointid_to_xyz,
            radius=8
        )

        if xyz is None:
            continue

        query_pts.append(qpt)
        world_pts.append(xyz)

    if len(query_pts) == 0:
        return None, None

    query_pts = np.asarray(query_pts)
    world_pts = np.asarray(world_pts)

    return query_pts, world_pts


def remove_duplicate_points(query_pts, world_pts):

    unique = {}

    for q, w in zip(query_pts, world_pts):

        key = tuple(np.round(w, 4))

        if key not in unique:
            unique[key] = (q, w)

    q_new = []
    w_new = []

    for q, w in unique.values():

        q_new.append(q)
        w_new.append(w)

    return np.asarray(q_new), np.asarray(w_new)

def estimate_pose_pnp(
        query_pts,
        world_pts,
        K):

    if len(query_pts) < 6:
        return None

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=world_pts.astype(np.float32),
        imagePoints=query_pts.astype(np.float32),
        cameraMatrix=K,
        distCoeffs=None,
        iterationsCount=10000,
        reprojectionError=8,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP
    )

    if not success:
        return None

    if inliers is not None and len(inliers) >= 6:

        cv2.solvePnPRefineLM(
            world_pts[inliers[:, 0]],
            query_pts[inliers[:, 0]],
            K,
            None,
            rvec,
            tvec
        )

    R, _ = cv2.Rodrigues(rvec)

    return R, tvec, inliers


def build_camera_matrix(camera):

    if camera.model == "PINHOLE":

        fx, fy, cx, cy = camera.params

    elif camera.model == "SIMPLE_PINHOLE":

        f, cx, cy = camera.params

        fx = f
        fy = f

    elif camera.model == "SIMPLE_RADIAL":

        f, cx, cy, k = camera.params

        fx = f
        fy = f

    else:
        raise NotImplementedError(camera.model)

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    return K

def compute_pose_error(
        R_pred,
        t_pred,
        R_gt,
        t_gt):

    t_error = np.linalg.norm(
        t_pred.reshape(3) -
        t_gt.reshape(3)
    )

    R_rel = R_pred @ R_gt.T

    cos_theta = (
        np.trace(R_rel) - 1
    ) / 2

    cos_theta = np.clip(
        cos_theta,
        -1,
        1
    )

    r_error = np.degrees(
        np.arccos(cos_theta)
    )

    return t_error, r_error


from scipy.spatial.transform import Rotation

def rotmat_to_qvec(R):
    """
    COLMAP/NVM quaternion format:
    qw qx qy qz
    """
    q = Rotation.from_matrix(R).as_quat()

    # scipy:
    # qx qy qz qw

    qx, qy, qz, qw = q

    return np.array([qw, qx, qy, qz])


def save_aachen_submission(
        results,
        output_file):

    with open(output_file, "w") as f:

        for name, qvec, t in results:

            qw, qx, qy, qz = qvec

            tx, ty, tz = t

            line = (
                f"{name} "
                f"{qw:.15f} "
                f"{qx:.15f} "
                f"{qy:.15f} "
                f"{qz:.15f} "
                f"{tx:.15f} "
                f"{ty:.15f} "
                f"{tz:.15f}\n"
            )

            f.write(line)
            

def find_all_query_images(root_dir):

    image_list = []

    for root, _, files in os.walk(root_dir):

        for f in files:

            if f.lower().endswith((".jpg", ".png")):

                image_list.append(
                    os.path.join(root, f)
                )

    image_list.sort()

    return image_list




    