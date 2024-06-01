# !/usr/bin/env python2
from math import pi, cos, sin, atan2, acos, sqrt, pow, radians, asin
from math_calc import *
from service_router import readPos


class LegConsts(object):
    ''' Class object to store characteristics of each leg '''
    def __init__(self, x_off, y_off, z_off, ang_off, leg_nr):
        self.x_off = x_off                # X offset from body origin to first servo (mm)
        self.y_off = y_off                # Y offset from body origin to first servo (mm)
        self.z_off = z_off                # Z offset from body origin to first servo (mm)
        self.ang_off = ang_off            # Angular offset from body origin to first servo (mm)
        self.f_ang_off = radians(13.33)   # Angular offset of Femur
        self.t_ang_off = radians(-25.90)  # Angular offset of Tibia
        self.c_len = 66.50                # Link length of Coxa  (mm)
        self.f_len = 144.40               # Link length of Femur (mm)
        self.t_len = 287                  # Link length of Tibia (mm)
        self.leg_nr = leg_nr              # Leg Number


class Kinematics(object):

    ''' Class object to compute various types of kinematics data for AntBot '''
    # Origin to coxa: x_off, y_off, z_off, ang_off, name
    leg1 = LegConsts(70.5, 122.225, -14.9, - pi / 3, "Leg 1")
    leg2 = LegConsts(-70.5, 122.225, -14.9, -2 * pi / 3, "Leg 2")
    leg3 = LegConsts(141.33, 0, -14.9, 0, "Leg 3")
    leg4 = LegConsts(-141.33, 0, -14.9, pi, "Leg 4")
    leg5 = LegConsts(70.5, -122.225, -14.9, pi / 3, "Leg 5")
    leg6 = LegConsts(-70.5, -122.225, -14.9, 2 * pi / 3, "Leg 6")
    leg_list = [leg1, leg2, leg3, leg4, leg5, leg6]

    ################
    # Public methods
    ################

    def doFkine(self, all_positions):
        ''' Function:  computes forward kinematics
            Parameter: all_positions: list with 18 values of servo positions in steps from ID1 to ID18
            Return:    ee_xyz: list of x,y,z coordinates for all 6 legs
                       servoPos: servo positions in radians
        '''
        servoPos = self.step_to_rad(all_positions)
        ee_xyz = []
        j = 0
        for i in xrange(0, 16, 3):
            ee_xyz.extend(self.calc_fkine(servoPos[i:i + 3], self.leg_list[j]))
            j += 1
        return ee_xyz, servoPos

    def doIkine(self, all_positions, x, y, z, body_orient=None, leg=None, auto=None):
        ''' Function:   computes inverse kinematics
            Parameters: all_positions: list with 18 values of servo positions in steps from ID1 to ID18;
                        x,y,z: desired change in x,y,z coordinates (same for all legs)
                        body_orient: list of 3 integers meaning alpha,beta,gamma rotation in degrees
                        leg: list with integers meaning leg numbers to compute inverse for them only
            Return:     list of 18 integers with servo steps
        '''
        ee_xyz, servoPos = self.doFkine(all_positions)
        thetas = []
        j = 0

        if isinstance(leg, int):
            leg = [leg]
        elif isinstance(leg, tuple):
            leg = list(leg)
        elif isinstance(body_orient, tuple):
            body_orient = list(body_orient)
        if body_orient:
            # Optional parameter. Compute inverse with body orientation
            body_orient = [radians(d) for d in body_orient]
            alpha_rad, beta_rad, gama_rad = body_orient[0], body_orient[1], body_orient[2]
            x = (cos(gama_rad) * sin(beta_rad) * z + sin(gama_rad) * sin(beta_rad) * y + x * cos(beta_rad)) \
                * cos(alpha_rad) - sin(alpha_rad) * (cos(gama_rad) * y - sin(gama_rad) * z)
            y = (cos(gama_rad) * sin(beta_rad) * z + sin(gama_rad) * sin(beta_rad) * y + x * cos(beta_rad)) \
                * sin(alpha_rad) + cos(alpha_rad) * (cos(gama_rad) * y - sin(gama_rad) * z)
            z = -sin(beta_rad) * x + cos(beta_rad) * sin(gama_rad) * y + cos(beta_rad) * cos(gama_rad) * z

        if leg:
            # Optional parameter. Compute inverse for a specific leg/s.
            for i in range(len(leg)):
                j = leg[i] - 1
                thetas.extend(self.calc_ikine(x, y, z, ee_xyz[j * 3:j * 3 + 3], self.leg_list[j]))

        else:
            # Compute inverse for all legs if not leg specified.
            for i in xrange(0, 16, 3):
                thetas.extend(self.calc_ikine(x, y, z, ee_xyz[i:i + 3], self.leg_list[j]))
                j += 1

        result = [int(each_theta) for each_theta in self.rad_to_step(thetas)]
        return result

    def doIkineRotationEuler(self, all_positions, alpha_rad, beta_rad, gama_rad, dist_x, dist_y, dist_z):
        ''' Function:   computes inverse kinematics and body rotation (Parallel kinematics)
            Parameters: all_positions: list with 18 values of servo positions in steps from ID1 to ID18;
                        alpha,beta,gama:    # for leg in range(6):  # 6 legs
    #     if leg in leg_list:
    #         new_pos.extend(K.calc_ikine(x, y, z, ee_xyz[leg:leg + 3], K.leg_list[leg]))
    #     else:
    #         new_pos.append(current_pos[3 * leg])
    #         new_pos.append(current_pos[3 * leg + 1])
    #         new_pos.append(current_pos[3 * leg + 2])ers with servo steps
        '''
        final_eexyz, ee_xyz = self.calc_rot_matrix(all_positions, alpha_rad, beta_rad, gama_rad)
        thetas = []
        j = 0
        for i in xrange(0, 16, 3):
            thetas.extend(self.calc_ikine(final_eexyz[i] - dist_x, final_eexyz[i + 1] - dist_y, final_eexyz[i + 2] - dist_z, ee_xyz[i:i + 3], self.leg_list[j]))
            j += 1
        result = [int(each_theta) for each_theta in self.rad_to_step(thetas)]
        return result

    def printForward(self, all_positions):
        ''' Function:   Prints x,y,z coordinates of each leg
            Parameters: all_positions: list with 18 values of servo positions in steps from ID1 to ID18;
        '''
        ee_list, theta_list = self.doFkine(all_positions)
        RoundedCoords = ['%.4f' % elem for elem in ee_list]
        print ""
        print "X,Y,Z coordinates of Leg end-points: "
        print "       " + str(["X       ", " Y    ", "  Z   "])
        print "Leg 1: " + str(RoundedCoords[0:3])
        print "Leg 2: " + str(RoundedCoords[3:6])
        print "Leg 3: " + str(RoundedCoords[6:9])
        print "Leg 4: " + str(RoundedCoords[9:12])
        print "Leg 5: " + str(RoundedCoords[12:15])
        print "Leg 6: " + str(RoundedCoords[15:18])
        print ""

    def printInverse(self, all_positions, x, y, z):
        ''' Function:   Prints servo positions, in radians, needed to reach the position
            Parameters: theta_list: 18 servo positions in radians.
        '''
        theta_list = self.doIkine(all_positions, x, y, z)
        RoundedThetas = ['%.4f' % elem for elem in theta_list]
        print ""
        print "Theta angles of each servo:"
        print "       " + str(["Coxa    ", "Femur ", "Tibia"])
        print "Leg 1: " + str(RoundedThetas[0:3])
        print "Leg 2: " + str(RoundedThetas[3:6])
        print "Leg 3: " + str(RoundedThetas[6:9])
        print "Leg 4: " + str(RoundedThetas[9:12])
        print "Leg 5: " + str(RoundedThetas[12:15])
        print "Leg 6: " + str(RoundedThetas[15:18])
        print ""

    def printKinematics(self, all_positions, x, y, z):
        self.printForward(all_positions)
        self.printInverse(all_positions, x, y, z)

    #################
    # Private methods
    #################

    def calc_fkine(self, servoPos, leg):
        theta1 = servoPos[0] - leg.ang_off
        theta2 = servoPos[1] + leg.f_ang_off
        theta3 = servoPos[2] + leg.t_ang_off
        ee_z = leg.f_len * sin(theta2) + leg.t_len * sin(theta3 + theta2) + leg.z_off
        ee_x = leg.x_off + cos(theta1) * (leg.c_len + leg.f_len * cos(theta2) + leg.t_len * cos(theta3 + theta2))
        ee_y = leg.y_off + sin(theta1) * (leg.c_len + leg.f_len * cos(theta2) + leg.t_len * cos(theta3 + theta2))
        return [ee_x, ee_y, ee_z]

    def calc_ikine(self, x, y, z, ee_xyz, leg, auto=None):
        init_X = ee_xyz[0]
        init_Y = ee_xyz[1]
        init_Z = ee_xyz[2]
        X = init_X + (x) - leg.x_off
        Y = init_Y + (y) - leg.y_off
        Z = init_Z + (z) - leg.z_off
        theta1 = atan2(Y, X) + leg.ang_off
        if theta1 < -pi:
            theta1 += 2 * pi
        if theta1 > pi:
            theta1 -= 2 * pi
        new_x = cos(leg.ang_off) * X - sin(leg.ang_off) * Y
        new_y = sin(leg.ang_off) * X + cos(leg.ang_off) * Y
        final_x = cos(theta1) * new_x + sin(theta1) * new_y - leg.c_len
        s = sqrt(pow(final_x, 2) + pow(Z, 2))
        try:
            t3_term = (-pow(s, 2) + pow(leg.f_len, 2) + pow(leg.t_len, 2)) / (2 * leg.f_len * leg.t_len)
            t3 = pi - acos(t3_term)
        except ValueError:
            print "Cannot compute acos(", t3_term, ") for ", leg.leg_nr
            if auto is None:
                if t3_term < 0:
                    t3 = pi - acos(-0.99)
                else:
                    t3 = pi - acos(0.99)
            else:
                return -1

        theta3 = -t3 - leg.t_ang_off
        theta2 = -(-atan2(Z, final_x) - atan2(leg.t_len * sin(t3), leg.f_len + leg.t_len * cos(t3)) + leg.f_ang_off)

        if auto is not None:
            if (theta2 > 1.8 or theta2 < -1.8) or (theta3 < -2.2 or theta3 > 2.2):
                return -1

        return [theta1, theta2, theta3]

    def calc_rot_displacement(self, alpha_rad, beta_rad, gama_rad, ee_xyz):
        pre_x = ee_xyz[0]
        pre_y = ee_xyz[1]
        pre_z = ee_xyz[2]
        r_term1 = (cos(gama_rad) * sin(beta_rad) * pre_z + sin(gama_rad) * sin(beta_rad) * pre_y + pre_x * cos(beta_rad))
        r_term2 = (cos(gama_rad) * pre_y - sin(gama_rad) * pre_z)
        r_x = r_term1 * cos(alpha_rad) - r_term2 * sin(alpha_rad) - pre_x
        r_y = r_term1 * sin(alpha_rad) + r_term2 * cos(alpha_rad) - pre_y
        r_z = - sin(beta_rad) * pre_x + cos(beta_rad) * sin(gama_rad) * pre_y + cos(beta_rad) * cos(gama_rad) * pre_z - pre_z
        return [r_x, r_y, r_z]

    def calc_rot_matrix(self, all_positions, alpha_rad, beta_rad, gama_rad):
        ee_xyz, servoPos = self.doFkine(all_positions)
        rot_val_list = []
        for i in xrange(0, 16, 3):
            rot_val_list.extend(self.calc_rot_displacement(alpha_rad, beta_rad, gama_rad, ee_xyz[i:i + 3]))
        return rot_val_list, ee_xyz

    def rad_to_step(self, pos_rads):
        return [i / pi * 2048 + 2048 for i in pos_rads]

    def step_to_rad(self, pos_steps):
        return [(((x / 2047.5) - 1) * pi) for x in pos_steps]

    def make_poligonCorners(self, all_positions, leg_list):
        if leg_list is int:
            leg_list = [leg_list]

        xyz_polygon = []
        ee_xyz, servoPos = self.doFkine(all_positions)
        newEe_xyz = [ee_xyz[0], ee_xyz[1], ee_xyz[2], ee_xyz[3], ee_xyz[4], ee_xyz[5],
                     ee_xyz[9], ee_xyz[10], ee_xyz[11], ee_xyz[15], ee_xyz[16], ee_xyz[17],
                     ee_xyz[12], ee_xyz[13], ee_xyz[14], ee_xyz[6], ee_xyz[7], ee_xyz[8]]

        for i in range(len(leg_list)):
            j = leg_list[i] - 1
            xyz_polygon.extend((newEe_xyz[j * 3:j * 3 + 3]))
        return xyz_polygon

    def make_polygonLines(self, leg_list, ee_xyz):
        print("leglistLins", leg_list)
        line = []
        for i in range(len(ee_xyz / 3)):
            j = i - 1
            line.extend = [ee_xyz[3 * j + 3] - ee_xyz[3 * j],
                           ee_xyz[3 * j + 4] - ee_xyz[3 * j + 1],
                           ee_xyz[3 * j + 5] - ee_xyz[3 * j + 2]]
        return line

    def check_stabilty(self, t_poly=None):
        ee_xyz, servoPos = self.doFkine(readPos())
        tac = [False, True, False, True, True, False]
        leg_list = []
        for i in range(len(tac)):
            if tac[i] is True:
                leg_list.extend([i + 1])
        poly_lines, poly_points = self.make_polygonLines(leg_list, ee_xyz)
        print("lines", poly_lines)
        if tac[1] is True and tac[2] is True and tac[5]is True:
            # gamma, beta = 10,20 #self.get_orientation(tac)
            # n = [0,-sin(beta),cos(beta)]
            print("im not here")
            P1 = [ee_xyz[3], ee_xyz[4], 1]
            P2 = [ee_xyz[6], ee_xyz[7], 1]
            P3 = [ee_xyz[15], ee_xyz[16], 1]
            print(P1, P2, P3)
        elif tac[0] is True and tac[3] is True and tac[4] is True:
            print("im here")
            P1 = [ee_xyz[0], ee_xyz[1], 1]
            P3 = [ee_xyz[9], ee_xyz[10], 1]
            P2 = [ee_xyz[12], ee_xyz[13], 1]
            print(P1, P2, P3)
        k = 1  # dotProduct(n,P1)
        x = 0
        y = 1
        z = 2
        lambda_1 = ((P2[x] * P3[y] - P2[y] * P3[x]) * k) / (P1[x] * P2[y] * P3[z] - P1[x] * P2[z] * P3[y] - P1[y] * P2[x] * P3[z] + P1[y] * P2[z] * P3[x] + P1[z] * P2[x] * P3[y] - P1[z] * P2[y] * P3[x])
        lambda_2 = -((P1[x] * P3[y] - P1[y] * P3[x]) * k) / (P1[x] * P2[y] * P3[z] - P1[x] * P2[z] * P3[y] - P1[y] * P2[x] * P3[z] + P1[y] * P2[z] * P3[x] + P1[z] * P2[x] * P3[y] - P1[z] * P2[y] * P3[x])
        lambda_3 = ((P1[x] * P2[y] - P1[y] * P2[x]) * k) / (P1[x] * P2[y] * P3[z] - P1[x] * P2[z] * P3[y] - P1[y] * P2[x] * P3[z] + P1[y] * P2[z] * P3[x] + P1[z] * P2[x] * P3[y] - P1[z] * P2[y] * P3[x])
        if lambda_1 > 0.1 and lambda_2 > 0.1 and lambda_3 > 0.1 and lambda_3 > 0.1:
            if lambda_1 < 0.9 and lambda_2 < 0.9 and lambda_3 < 0.9:
                if lambda_1 + lambda_2 + lambda_3 == 1:
                    inside = True
        side1 = subtract(P1, P2)
        side2 = subtract(P3, P2)
        side3 = subtract(P1, P3)
        G = [0, 0, 1]
        P2_G = subtract(G, P2)
        P3_G = subtract(G, P3)
        margin_s1 = sqrt(pow(dotProduct(P2_G, unit_vec(side1)), 2) + dotProduct(P2_G, P2_G))
        margin_s2 = sqrt(pow(dotProduct(P2_G, unit_vec(side2)), 2) + dotProduct(P2_G, P2_G))
        margin_s3 = sqrt(pow(dotProduct(P3_G, unit_vec(side3)), 2) + dotProduct(P3_G, P3_G))
        stability_margin = min(margin_s1, margin_s2, margin_s3)
        print(stability_margin, inside)
        return stability_margin, inside

    def get_orientation(self, leg_list):
        ee_xyz, servoPos = self.doFkine(readPos())
        p1 = ee_xyz[3 * (leg_list[0] - 1):3 * (leg_list[0] - 1) + 3]
        p2 = ee_xyz[3 * (leg_list[1] - 1):3 * (leg_list[1] - 1) + 3]
        p3 = ee_xyz[3 * (leg_list[2] - 1):3 * (leg_list[2] - 1) + 3]
        p21 = subtract(p2, p1)
        p23 = subtract(p2, p3)
        normz = crossProduct(p21, p23)
        beta = atan2(normz[0], normz[2]) * 180 / pi
        gamma = -atan2(normz[1], normz[2]) * 180 / pi
        return gamma, beta

    def calc_translationStairs(self, riser, climbed_stairs_front, climbed_stairs_rear):
        # gamma, beta = self.get_orientation([1,5,6])
        ee_xyz, servopos = self.doFkine(readPos())
        dist_y = abs(ee_xyz[1] - ee_xyz[13])
        riser_diff = (climbed_stairs_front - climbed_stairs_rear) * riser
        omega = asin(riser_diff / dist_y) * 180 / pi
        AB = -ee_xyz[14] + 30
        AC = AB / cos(omega * pi / 180)
        BC = AC * sin(omega * pi / 180)
        BE = sqrt(pow(ee_xyz[12], 2) + pow(ee_xyz[11], 2)) - 141.33
        CE = BE - BC
        CD = BC * CE / AC

        if AC + CD <= riser_diff:
            trans_z_g = riser_diff - AC - CD + 10
            translation_z = trans_z_g * cos(omega * pi / 180)
            translation_y = trans_z_g * sin(omega * pi / 180)
        else:
            translation_z = 0
            translation_y = 0
        return [translation_z, translation_y]
