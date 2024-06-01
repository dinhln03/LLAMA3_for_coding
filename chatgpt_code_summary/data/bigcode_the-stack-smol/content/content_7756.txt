#!/usr/bin/env python

import os
import cv2
import numpy as np
from enum import Enum
import math

class Calc (Enum):
    OPENCV = 1
    GSL_MULTI_ROOT = 2
    GSL_MULTI_FIT = 3

image_file_name = "Man2_10deg.png"

use_calc = Calc.GSL_MULTI_FIT
#use_calc = Calc.GSL_MULTI_ROOT
#use_calc = Calc.OPENCV

def get_project_xy(A, R, X, Y, Z):
    P = np.array([X, Y, Z, 1])
    pp = A.dot(R.dot(P))
    return [pp[0]/pp[2], pp[1]/pp[2]]

def get_project_uv(A, R, X, Y, Z):
    fx, fy, cx, cy = A[0][0], A[1][1], A[0][2], A[1][2]
    r11, r12, r13, t1 = R[0][0], R[0][1], R[0][2], R[0][3]
    r21, r22, r23, t2 = R[1][0], R[1][1], R[1][2], R[1][3]
    r31, r32, r33, t3 = R[2][0], R[2][1], R[2][2], R[2][3]
    s = r31 * X + r32 * Y + r33 * Z + t3
#    print("%f * %f + %f * %f + %f * %f + %f = %f\n" % (r31, X, r32, Y, r33, Z, t3, s))
    u = ((fx*r11 + cx*r31)*X + (fx*r12 + cx*r32)*Y + (fx*r13 + cx*r33)*Z + fx*t1 + cx*t3)/s
    v = ((fy*r21 + cy*r31)*X + (fy*r22 + cy*r32)*Y +(fy*r23 + cy*r33)*Z + fy*t2 + cy*t3)/s
#    print("%f/%f" % ((fx*r11 + cx*r31)*X + (fx*r12 + cx*r32)*Y + (fx*r13 + cx*r33)*Z + fx*t1 + cx*t3, s))
#    print("%f/%f" % ((fy*r21 + cy*r31)*X + (fy*r22 + cy*r32)*Y +(fy*r23 + cy*r33)*Z + fy*t2 + cy*t3, s))
    return u, v


def get_rot_tran_matrix2(M):
    a = []
    for i in range(0, 12):
        a.append(float(M[i]))
    R = np.array([[a[0], a[1], a[2], a[9]], [a[3], a[4], a[5], a[10]], [a[6], a[7], a[8], a[11]]])
    return R

def print_rotation_angle(RT):
    R = RT[:, 0:3]
#    print('R:', R)
    V = R.dot(np.array([0, 0, 1]))
    print('\033[92mV:', V)
    print('phi = %f degree' % math.degrees(math.atan(V[0] / V[2])))
    print('theta = %f degree' % (math.sqrt(V[0]**2 + V[2]**2)))

    print('\033[0m')


def verification_rot_tran_matrix(A, R, u, v, X, Y, Z):
    P = np.array([X, Y, Z, 1], dtype="double")
    pp = A.dot(R.dot(P))
    diff = np.fabs(u - pp[0]/pp[2]) + np.fabs(v - pp[1]/pp[2])
    print(u, v, '<->', pp[0]/pp[2], pp[1]/pp[2])
    return diff

def verification_rot_tran_matrix_2(A, R, u, v, X, Y, Z):
    ud, vd = get_project_uv(A, R, X, Y, Z)
    print(u, v, '<->', ud, vd)

def get_rot_tran_matrix(img_pnts, mod_pnts, cam_matrix): # s = 1
    (u1, v1) = img_pnts[0]  # nose tip
    (u2, v2) = img_pnts[1]  # left eye
    (u3, v3) = img_pnts[2]  # right eye
    (u4, v4) = img_pnts[3]  # left mouth
    (u5, v5) = img_pnts[4]  # right mouth
    (X1, Y1, Z1) = model_points[0]
    (X2, Y2, Z2) = model_points[1]
    (X3, Y3, Z3) = model_points[2]
    (X4, Y4, Z4) = model_points[3]
    (X5, Y5, Z5) = model_points[4]
    fx = cam_matrix[0][0]
    fy = cam_matrix[1][1]
    cx = cam_matrix[0][2]
    cy = cam_matrix[1][2]
    r31, r32, r33, t3 = 0, 0, 0, 1
    D = np.array([[X1, Y1, Z1, 1], [X2, Y2, Z2, 1], [X3, Y3, Z3, 1], [X4, Y4, Z4, 1]])
    D1 = np.array([[(v1 - cy) / fy, Y1, Z1, 1], [(v2 - cy) / fy, Y2, Z2, 1], [(v3 - cy) / fy, Y3, Z3, 1],
                  [(v4 - cy) / fy, Y4, Z4, 1]])
    D2 = np.array([[X1, (v1 - cy) / fy, Z1, 1], [X2, (v2 - cy) / fy, Z2, 1], [X3, (v3 - cy) / fy, Z3, 1],
                  [X4, (v4 - cy) / fy, Z4, 1]])
    D3 = np.array([[X1, Y1, (v1 - cy) / fy, 1], [X2, Y2, (v2 - cy) / fy, 1], [X3, Y3, (v3 - cy) / fy, 1],
                  [X4, Y4, (v4 - cy) / fy, 1]])
    D4 = np.array([[X1, Y1, Z1, (v1 - cy) / fy], [X2, Y2, Z2, (v2 - cy) / fy], [X3, Y3, Z3, (v3 - cy) / fy],
                  [X4, Y4, Z4, (v4 - cy) / fy]])
    r21 = np.linalg.det(D1) / np.linalg.det(D)
    r22 = np.linalg.det(D2) / np.linalg.det(D)
    r23 = np.linalg.det(D3) / np.linalg.det(D)
    t2 = np.linalg.det(D4) / np.linalg.det(D)
    D1 = np.array([[(u1 - cx) / fx, Y1, Z1, 1], [(u2 - cx) / fx, Y2, Z2, 1], [(u3 - cx) / fx, Y3, Z3, 1],
                  [(u4 - cx) / fx, Y4, Z4, 1]])
    D2 = np.array([[X1, (u1 - cx) / fx, Z1, 1], [X2, (u2 - cx) / fx, Z2, 1], [X3, (u3 - cx) / fx, Z3, 1],
                  [X4, (u4 - cx) / fx, Z4, 1]])
    D3 = np.array([[X1, Y1, (u1 - cx) / fx, 1], [X2, Y2, (u2 - cx) / fx, 1], [X3, Y3, (u3 - cx) / fx, 1],
                  [X4, Y4, (u4 - cx) / fx, 1]])
    D4 = np.array([[X1, Y1, Z1, (u1 - cx) / fx], [X2, Y2, Z2, (v2 - cy) / fy], [X3, Y3, Z3, (u3 - cx) / fx],
                  [X4, Y4, Z4, (u4 - cx) / fx]])
    r11 = np.linalg.det(D1) / np.linalg.det(D)
    r12 = np.linalg.det(D2) / np.linalg.det(D)
    r13 = np.linalg.det(D3) / np.linalg.det(D)
    t1 = np.linalg.det(D4) / np.linalg.det(D)
    R = np.array([[r11, r12, r13, t1], [r21, r22, r23, t2], [r31, r32, r33, t3]])
    return R


if __name__ == '__main__':
    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (-50.0, -40.0, 20.0),  # Left eye left corner
        (50.0, -40.0, 20.0),  # Right eye right corner
        (-27.5, 30.0, 10.0),  # Left Mouth corner
        (27.5, 30.0, 10.0)  # Right mouth corner
    ])
    index = 4
    points_file = "points.txt"
    image_file = []
    key_points = []
    matrix = []
    if not os.path.exists(points_file):
        print('do not have file %s' % points_file)
        exit(0)
    points_f = open(points_file, 'r')

    for line in points_f:
        a = line.split('|')
        b = a[0].split(',')
        image_file.append(b[0])
        key_points.append(b[1:11])
        matrix.append(a[1].split(','))

    points_f.close()

    image_points = np.array([
        (int(key_points[index][0]), int(key_points[index][5])),  # Nose tip
        (int(key_points[index][1]), int(key_points[index][6])),  # Left eye left corner
        (int(key_points[index][2]), int(key_points[index][7])),  # Right eye right corner
        (int(key_points[index][3]), int(key_points[index][8])),  # Left Mouth corner
        (int(key_points[index][4]), int(key_points[index][9]))  # Right mouth corner
    ], dtype="double")

    # Read Image
    im = cv2.imread(image_file[index])
    size = im.shape

    # Camera internals
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double")

    R = get_rot_tran_matrix2(matrix[index])  # read gsl result

    print("\033[94m----check----")
    for i in range(0, 5):
        verification_rot_tran_matrix_2(camera_matrix, R, image_points[i][0], image_points[i][1],
                                 model_points[i][0], model_points[i][1], model_points[i][2])
    print("----end-----\033[0m")

    print_rotation_angle(R)
    print("rotation_matrix:\n {0}".format(R))

    # draw axes
    axis_length = 100.0

    if False:
        Z_pnt = get_project_uv(camera_matrix, R, 0, 0, axis_length)
        Y_pnt = get_project_uv(camera_matrix, R, 0, axis_length, 0)
        X_pnt = get_project_uv(camera_matrix, R, axis_length, 0, 0)
        Org_pnt = get_project_uv(camera_matrix, R, 0, 0, 0)
    else:
        Z_pnt = get_project_xy(camera_matrix, R, 0, 0, axis_length)
        Y_pnt = get_project_xy(camera_matrix, R, 0, axis_length, 0)
        X_pnt = get_project_xy(camera_matrix, R, axis_length, 0, 0)
        Org_pnt = get_project_xy(camera_matrix, R, 0, 0, 0)


    #print('Rt:\033[93m', R, '\033[0m')
#    print('X:\033[93m', R[:, 0:3].dot(np.array([axis_length, 0, 0])), '\033[0m')
#    print('Y:\033[93m', R[:, 0:3].dot(np.array([0, axis_length, 0])), '\033[0m')
#    print('Z:\033[93m', R[:, 0:3].dot(np.array([0, 0, axis_length])), '\033[0m')

    for p in image_points:
        cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

    p1 = (int(Org_pnt[0]), int(Org_pnt[1]))
    p2 = (int(Z_pnt[0]), int(Z_pnt[1]))
    cv2.line(im, p1, p2, (255, 0, 0), 2)  #blue:Z
    p1 = (int(Org_pnt[0]), int(Org_pnt[1]))
    p2 = (int(Y_pnt[0]), int(Y_pnt[1]))
    cv2.line(im, p1, p2, (0, 255, 0), 2)  #green:Y
    p1 = (int(Org_pnt[0]), int(Org_pnt[1]))
    p2 = (int(X_pnt[0]), int(X_pnt[1]))
    cv2.line(im, p1, p2, (0, 255, 255), 2)  #yellow: X
    # Display image
    cv2.imshow("Output", im)
    cv2.waitKey(0)
