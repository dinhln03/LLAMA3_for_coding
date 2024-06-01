#!/usr/bin/env python
import argparse
import open3d as o3d
import numpy as np
import os
import time
from os.path import join, dirname, basename, splitext, exists, isdir, isfile
from os import listdir
from numpy import linalg as LA
import math
import cv2
from pathlib import Path


def pcd_to_bin(pcd_path, outdir=None):
    pcd = o3d.io.read_point_cloud(pcd_path, format="pcd")
    pcd_arr = np.asarray(pcd.points)

    if len(pcd_arr) == 0:
        return None

    outpath = join(Path(pcd_path).parent if outdir is None else outdir, splitext(basename(pcd_path))[0] + ".bin")

    # binarize array and save to the same file path with .bin extension
    pcd_arr.tofile(outpath)
    return outpath


def pcd_to_sphproj(pcd_path, nr_scans, width, outdir=None):
    pcd = o3d.io.read_point_cloud(pcd_path, format="pcd")
    pcd_arr = np.asarray(pcd.points)

    if len(pcd_arr) == 0:
        return None

    # https://towardsdatascience.com/spherical-projection-for-point-clouds-56a2fc258e6c

    # print(pcd_arr.shape)
    # print(pcd_arr[:, :3].shape)

    R = LA.norm(pcd_arr[:, :3], axis=1)
    print("R {} | {} -- {}".format(R.shape, np.amin(R), np.amax(R)))

    yaw = np.arctan2(pcd_arr[:, 1], pcd_arr[:, 0])
    # print("yaw {} | {} -- {}".format(yaw.shape, np.amin(yaw), np.amax(yaw)))
    # print("y {} | {} -- {}".format(pcd_arr[:, 1].shape, np.amin(pcd_arr[:, 1]), np.amax(pcd_arr[:, 1])))

    pitch = np.arcsin(np.divide(pcd_arr[:, 2], R))
    # print("pitch {} | {} -- {}".format(pitch.shape, np.amin(pitch), np.amax(pitch)))

    # import matplotlib.pyplot as plt
    # plt.plot(yaw, pitch, 'b.')
    # plt.xlabel('yaw [rad]')
    # plt.ylabel('pitch [rad]')
    # plt.axis('equal')
    # plt.show()

    FOV_Down = np.amin(pitch)
    FOV_Up = np.amax(pitch)
    FOV = FOV_Up + abs(FOV_Down)

    u = np.around((nr_scans-1) * (1.0-(pitch-FOV_Down)/FOV)).astype(np.int16)
    # print("u {} | {} -- {} | {}".format(u.shape, np.amin(u), np.amax(u), u.dtype))

    v = np.around((width-1) * (0.5 * ((yaw/math.pi) + 1))).astype(np.int16)
    # print("v {} | {} -- {} | {}".format(v.shape, np.amin(v), np.amax(v), v.dtype))

    sph_proj = np.zeros((nr_scans, width))

    R[R > 100.0] = 100.0 # cut off all values above 100m
    R = np.round((R / 100.0) * 255.0)  # convert 0.0-100.0m into 0.0-255.0 for saving as byte8 image

    sph_proj[u, v] = R
    # print("sph_proj {} | {} -- {} | {}".format(sph_proj.shape, np.amin(sph_proj), np.amax(sph_proj), sph_proj.dtype))

    outpath = join(Path(pcd_path).parent if outdir is None else outdir, splitext(basename(pcd_path))[0] + ".jpg")
    cv2.imwrite(outpath, sph_proj)
    print(outpath)
    return np.amin(R), np.amax(R)


def bin_to_pcd(bin_path, outdir=None):
    print(bin_path)
    pcd_arr = np.fromfile(bin_path, dtype=np.float32)
    pcd_arr = pcd_arr.reshape((-1, 4))  # kitti has 4 values per point
    # print(type(pcd_arr), pcd_arr.shape, len(pcd_arr))
    # print(pcd_arr[:, :3].shape)
    if len(pcd_arr) == 0:
        return None

    outpath = join(Path(bin_path).parent if outdir is None else outdir, splitext(basename(bin_path))[0] + ".pcd")
    print(outpath)
    # save array as .pcd
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_arr[:, :3])  # 3 dimensions
    o3d.io.write_point_cloud(outpath, pcd)
    return outpath


def bin_to_sphproj(bin_path, outdir=None):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert between .pcd and .bin point cloud formats')
    parser.add_argument("-t", type=str, required=True,
                        help="Conversion to run (pcd2bin, pcd2sphproj, bin2pcd, bin2sphproj)")
    parser.add_argument("-p", type=str, required=True, help="Path to directory or file with point cloud")
    parser.add_argument("-nr_scans", type=int, help="Number of lidar scans (default 16)", default=16)
    parser.add_argument("-width", type=int, help="Spherical projection width (default 1024)", default=1024)

    args = parser.parse_args()

    if not exists(args.p):
        exit("{} does not exist".format(args.p))

    if isfile(args.p):
        # check extension
        ext = splitext(args.p)[-1].lower()

        if args.t == "pcd2bin" and ext == ".pcd":
            pcd_to_bin(args.p)
        elif args.t == "bin2pcd" and ext == ".bin":
            bin_to_pcd(args.p)
        elif args.t == "pcd2sphproj" and ext == ".pcd":
            pcd_to_sphproj(args.p, args.nr_scans, args.width)
        elif args.t == "bin2sphproj" and ext == ".bin":
            bin_to_sphproj(args.p)
        else:
            print("Wrong conversion or extension incompatible with conversion")

    elif isdir(args.p):
        # go through all files and convert .pcd or .bin files encountered within the directory

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        outdir = join(Path(args.p).parent, str(args.t) + "_" + timestamp)

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        range_min = float('inf')
        range_max = float('-inf')

        for f in listdir(args.p):
            # check extension
            ext = splitext(f)[-1].lower()

            if args.t == "pcd2bin" and ext == ".pcd":
                pcd_to_bin(join(args.p, f), outdir)
            elif args.t == "bin2pcd" and ext == ".bin":
                bin_to_pcd(join(args.p, f), outdir)
            elif args.t == "pcd2sphproj" and ext == ".pcd":
                range_min1, range_max1  = pcd_to_sphproj(join(args.p, f), args.nr_scans, args.width, outdir)
                if range_min1 < range_min:
                    range_min = range_min1
                if range_max1 > range_max:
                    range_max = range_max1
            elif args.t == "bin2sphproj" and ext == ".bin":
                bin_to_sphproj(join(args.p, f), outdir)
            else:
                print("Wrong conversion or extension incompatible with conversion")

        print("range: {} - {}".format(range_min, range_max))