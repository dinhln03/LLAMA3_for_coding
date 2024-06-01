import os
import numpy as np
from glob import glob
from scipy import optimize, spatial, ndimage
from tifffile import imread, imsave
from skimage.segmentation import find_boundaries
from skimage.morphology import remove_small_objects
from skimage.draw import line
from utils import random_colormap
import pdb

# define binarization function
def prepare_binary(fn):
    # generate binary segmentaiton result
    seg = np.squeeze(imread(fn)) > bw_th
    seg = remove_small_objects(seg>0, min_size=min_obj_size)
    return seg

# params
max_matching_dist = 45
approx_inf = 65535
track_display_legnth = 20
min_obj_size = 20
bw_th = -0.5

parent_path = "/mnt/data/"
all_movies = glob(parent_path + "timelapse/*.tiff")
for M_idx, movies in enumerate(all_movies):
    movie_basename = os.path.basename(movies)
    well_name = movie_basename[:-5]

    seg_path = f"{parent_path}timelapse_seg/{well_name}/"
    # vis_path = f"{parent_path}timelapse_track/{well_name}"
    # os.makedirs(vis_path, exist_ok=True)
    raw_path = f"{parent_path}timelapse/{well_name}"
    track_result = f"{parent_path}timelapse_track/{well_name}_result.npy"


    total_time = len(glob(raw_path + "/*.tiff"))
    traj = dict()
    lineage = dict()
    for tt in range(total_time):
        seg_fn = seg_path + f"img_{tt}_segmentation.tiff"

        seg = prepare_binary(seg_fn)

        # get label image
        seg_label, num_cells = ndimage.label(seg)

        # calculate center of mass
        centroid = ndimage.center_of_mass(seg, labels=seg_label, index=np.arange(1, num_cells + 1))

        # generate cell information of this frame
        traj.update({
            tt : {"centroid": centroid, "parent": [], "child": [], "ID": []}
        })

        
    # initialize trajectory ID, parent node, track pts for the first frame
    max_cell_id =  len(traj[0].get("centroid"))
    traj[0].update(
        {"ID": np.arange(0, max_cell_id, 1)}
    )
    traj[0].update(
        {"parent": -1 * np.ones(max_cell_id, dtype=int)}
    )
    centers = traj[0].get("centroid")
    pts = []
    for ii in range(max_cell_id):
        pts.append([centers[ii]])
        lineage.update({ii: [centers[ii]]})
    traj[0].update({"track_pts": pts})

    for tt in np.arange(1, total_time):
        p_prev = traj[tt-1].get("centroid")
        p_next = traj[tt].get("centroid")

        ###########################################################
        # simple LAP tracking
        ###########################################################
        num_cell_prev = len(p_prev)
        num_cell_next = len(p_next)

        # calculate distance between each pair of cells
        cost_mat = spatial.distance.cdist(p_prev, p_next)

        # if the distance is too far, change to approx. Inf.
        cost_mat[cost_mat > max_matching_dist] = approx_inf

        # add edges from cells in previous frame to auxillary vertices
        # in order to accomendate segmentation errors and leaving cells
        cost_mat_aug = max_matching_dist * 1.2 * np.ones(
            (num_cell_prev, num_cell_next + num_cell_prev), dtype=float
        )
        cost_mat_aug[:num_cell_prev, :num_cell_next] = cost_mat[:, :]

        # solve the optimization problem
        row_ind, col_ind = optimize.linear_sum_assignment(cost_mat_aug)

        #########################################################
        # parse the matching result
        #########################################################
        prev_child = np.ones(num_cell_prev, dtype=int)
        next_parent = np.ones(num_cell_next, dtype=int)
        next_ID = np.zeros(num_cell_next, dtype=int)
        next_track_pts = []

        # assign child for cells in previous frame
        for ii in range(num_cell_prev):
            if col_ind[ii] >= num_cell_next:
                prev_child[ii] = -1
            else:
                prev_child[ii] = col_ind[ii]

        # assign parent for cells in next frame, update ID and track pts
        prev_pt = traj[tt-1].get("track_pts")
        prev_id = traj[tt-1].get("ID")
        for ii in range(num_cell_next):
            if ii in col_ind:
                # a matched cell is found
                next_parent[ii] = np.where(col_ind == ii)[0][0]
                next_ID[ii] = prev_id[next_parent[ii]]
                
                current_pts = prev_pt[next_parent[ii]].copy()
                current_pts.append(p_next[ii])
                if len(current_pts) > track_display_legnth:
                    current_pts.pop(0)
                next_track_pts.append(current_pts)
                # attach this point to the lineage
                single_lineage = lineage.get(next_ID[ii])
                try:
                    single_lineage.append(p_next[ii])
                except Exception:
                    pdb.set_trace()
                lineage.update({next_ID[ii]: single_lineage})
            else:
                # a new cell
                next_parent[ii] = -1
                next_ID[ii] = max_cell_id
                next_track_pts.append([p_next[ii]])
                lineage.update({max_cell_id: [p_next[ii]]})
                max_cell_id += 1

        # update record
        traj[tt-1].update({"child": prev_child})
        traj[tt].update({"parent": next_parent})
        traj[tt].update({"ID": next_ID})
        traj[tt].update({"track_pts": next_track_pts})

    np.save(track_result, [traj, lineage])

"""
######################################################
# generate track visualization
######################################################
cmap = random_colormap()
for tt in range(total_time):
    # print(traj[tt].get("ID"))

    # load segmentation and extract contours
    seg_fn = seg_path + f"img_{tt}_segmentation.tiff"
    seg = prepare_binary(seg_fn)
    seg_label, num_cells = ndimage.label(seg)
    cell_contours = find_boundaries(seg, mode='inner').astype(np.uint16)
    cell_contours[cell_contours > 0] = 1
    cell_contours = cell_contours * seg_label.astype(np.uint16)
    cell_contours = cell_contours - 1  # to make the first object has label 0, to match index

    # load raw image and create visualizaiton in RGB
    # TODO: use real raw images
    # raw = seg.astype(np.uint8)
    raw = np.squeeze(imread(raw_path + f"img_{tt}.tiff")).astype(np.float32)
    raw = (raw - raw.min())/ (raw.max() - raw.min())
    raw = raw * 255
    raw = raw.astype(np.uint8)
    vis = np.zeros((raw.shape[0], raw.shape[1], 3), dtype=np.uint8)
    for cc in range(3):
        vis[:, :, cc] = raw

    # loop through all cells, for each cell, we do the following
    # 1- find ID, 2- load the color, 3- draw contour 4- draw track
    cell_id = traj[tt].get("ID")
    pts = traj[tt].get("track_pts")
    for cid in range(num_cells):
        # find ID
        this_id = cell_id[cid]

        # load the color
        this_color = 255 * cmap.colors[this_id]
        this_color = this_color.astype(np.uint8)

        # draw contour
        for cc in range(3):
            vis_c = vis[:, :, cc]
            vis_c[cell_contours == cid] = this_color[cc]
            vis[:, :, cc] = vis_c  # TODO: check if we need this line

        # draw track
        this_track = pts[cid]

        if len(this_track) < 2:
            continue
        else:
            for pid in range(len(this_track) - 1):
                p1 = this_track[pid]
                p2 = this_track[pid + 1]
                rr, cc = line(int(round(p1[0])), int(round(p1[1])), int(round(p2[0])), int(round(p2[1])))
                for ch in range(3):
                    vis[rr, cc ,ch] = this_color[ch]

    imsave(vis_path + f"img_{tt+1}.tiff", vis)
"""
