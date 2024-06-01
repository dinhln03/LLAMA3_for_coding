import math

import numpy as np

from kinematics.forward import ForwardKinematics
from kinematics.kinematics import Kinematics
from kinematics.solution import InverseKinematicsShoulderSolution, InverseKinematicsSpecificSolution, \
    InverseKinematicsSolution, InverseKinematicsWristSolution


class InverseKinematics(Kinematics):
    def __init__(self):
        super().__init__()
        self.forward_kinematics = ForwardKinematics()

    def __clamp_cos_sin_within_threshold(self, cos_or_sin):
        new_val = cos_or_sin
        if 1 < new_val <= 1.2:
            new_val = 1.0
        elif -1.2 <= new_val < -1:
            new_val = -1.0

        return new_val

    def __compute_solution_for_theta_1(self, T06, theta_1, debug=False):
        wrist_solution = InverseKinematicsWristSolution()

        # Theta 5
        P06 = T06[:, 3]
        theta_5_1 = None
        theta_5_2 = None

        theta_5_cos = (P06[0] * math.sin(theta_1) - P06[1] * np.cos(
            theta_1) - self.joint4_dh.d) / self.joint6_dh.d

        theta_5_cos = self.__clamp_cos_sin_within_threshold(theta_5_cos)

        if -1 <= theta_5_cos <= 1:
            theta_5_1 = math.acos(theta_5_cos)
            theta_5_2 = -math.acos(theta_5_cos)

        sigma = 0.00001

        if theta_5_1 is not None and not -sigma <= math.sin(theta_5_1) <= sigma:
            wrist_solution.solution_wrist_up = self.__compute_solution_for_wrist(theta_1, theta_5_1, T06)
        else:
            wrist_solution.solution_wrist_up.is_valid_solution = False
        if theta_5_2 is not None and not -sigma <= math.sin(theta_5_2) <= sigma:
            wrist_solution.solution_wrist_down = self.__compute_solution_for_wrist(theta_1, theta_5_2, T06)
        else:
            wrist_solution.solution_wrist_down.is_valid_solution = False
        if not wrist_solution.solution_wrist_up.is_valid_solution and not wrist_solution.solution_wrist_down.is_valid_solution:
            wrist_solution.is_valid_solution = False

        if debug:
            print(f"Theta 5: {theta_5_1:.3f}, {theta_5_2:.3f}")

        return wrist_solution

    def __compute_solution_for_wrist(self, theta_1, theta_5, T06, debug=False):
        shoulder_solution = InverseKinematicsShoulderSolution()

        # Theta 6
        T60 = np.linalg.inv(T06)
        X60 = T60[:, 0]
        Y60 = T60[:, 1]

        theta_6_cos = (X60[0] * math.sin(theta_1) - Y60[0] * math.cos(theta_1)) / math.sin(
            theta_5)  # only using one of the theta 5's for now..
        theta_6_sin = (-X60[1] * math.sin(theta_1) + Y60[1] * math.cos(theta_1)) / math.sin(
            theta_5)  # only using one of the theta 5's for now..
        theta_6 = math.atan2(theta_6_sin, theta_6_cos)

        if debug:
            print(f"Theta 6: {theta_6:.3f}")

        tm_dict = {}

        # Theta 3
        T01 = self.compute_transformation_matrix(theta_1, self.joint1_dh)
        T45 = self.compute_transformation_matrix(theta_5, self.joint5_dh)
        T56 = self.compute_transformation_matrix(theta_6, self.joint6_dh)
        T46 = np.matmul(T45, T56)
        T64 = np.linalg.inv(T46)
        T10 = np.linalg.inv(T01)
        T14 = np.matmul(np.matmul(T10, T06), T64)
        P14 = T14[:, 3]

        tm_dict["T06"] = T06
        tm_dict["T01"] = T01
        tm_dict["T45"] = T45
        tm_dict["T56"] = T56
        tm_dict["T64"] = T64
        tm_dict["T10"] = T10
        tm_dict["T14"] = T14
        tm_dict["P14"] = P14

        theta_3_cos = (math.sqrt(
            P14[0] ** 2 + P14[2] ** 2) ** 2 - self.joint3_dh.a ** 2 - self.joint4_dh.a ** 2) / (
                                  2 * (-self.joint3_dh.a) * (-self.joint4_dh.a))
        if debug:
            print("theta3_cos: ", theta_3_cos)

        theta_3_cos = self.__clamp_cos_sin_within_threshold(theta_3_cos)

        if not -1 <= theta_3_cos <= 1:
            shoulder_solution.is_valid_solution = False
            return shoulder_solution

        theta_3_up = math.acos(theta_3_cos)
        theta_3_down = -math.acos(theta_3_cos)

        if debug:
            print(f"Theta 3: Up: {theta_3_up:.3f} Down: {theta_3_down:.3f}")

        shoulder_solution.solution_elbow_up = self.__compute_specific_solution(theta_1, theta_3_up, theta_5, theta_6, tm_dict)
        shoulder_solution.solution_elbow_down = self.__compute_specific_solution(theta_1, theta_3_down, theta_5, theta_6, tm_dict)

        return shoulder_solution

    def __compute_specific_solution(self, theta_1, theta_3, theta_5, theta_6, tm_dict, debug=False):
        specific_solution = InverseKinematicsSpecificSolution()

        P14 = tm_dict["P14"]

        phi_1 = math.atan2(-P14[2], -P14[0])
        phi_2 = math.asin((-self.joint4_dh.a * math.sin(theta_3)) / math.sqrt(P14[0]**2 + P14[2]**2))
        theta_2 = phi_1 - phi_2

        if debug:
            print(f"Theta 2: {theta_2:.3f}")

        T01 = tm_dict["T01"]
        T12 = self.compute_transformation_matrix(theta_2, self.joint2_dh)
        T23 = self.compute_transformation_matrix(theta_3, self.joint3_dh)
        T45 = tm_dict["T45"]
        T56 = tm_dict["T56"]
        T06 = tm_dict["T06"]
        T03 = np.matmul(np.matmul(T01, T12), T23)
        T30 = np.linalg.inv(T03)
        T64 = tm_dict["T64"]
        T34 = np.matmul(np.matmul(T30, T06), T64)
        X34 = T34[:, 0]

        theta_4 = math.atan2(X34[1], X34[0])

        if debug:
            print(f"Theta 4: {theta_4:.3f}")

        specific_solution.thetas = [theta_1, theta_2, theta_3, theta_4, theta_5, theta_6]

        return specific_solution

    def __print_all_solutions(self, solution):
        print("Inverse Solutions:")
        if solution.solution_shoulder_left.is_valid_solution:
            if solution.solution_shoulder_left.solution_wrist_up.is_valid_solution:
                if solution.solution_shoulder_left.solution_wrist_up.solution_elbow_up.is_valid_solution:
                    print(
                        f"Shoulder left, wrist up, elbow up: {solution.solution_shoulder_left.solution_wrist_up.solution_elbow_up.thetas}")
                if solution.solution_shoulder_left.solution_wrist_up.solution_elbow_down.is_valid_solution:
                    print(
                        f"Shoulder left, wrist up, elbow down: {solution.solution_shoulder_left.solution_wrist_up.solution_elbow_down.thetas}")
            if solution.solution_shoulder_left.solution_wrist_down.is_valid_solution:
                if solution.solution_shoulder_left.solution_wrist_down.solution_elbow_up.is_valid_solution:
                    print(
                        f"Shoulder left, wrist down, elbow up: {solution.solution_shoulder_left.solution_wrist_down.solution_elbow_up.thetas}")
                if solution.solution_shoulder_left.solution_wrist_down.solution_elbow_down:
                    print(
                        f"Shoulder left, wrist down, elbow down: {solution.solution_shoulder_left.solution_wrist_down.solution_elbow_down.thetas}")
        if solution.solution_shoulder_right.is_valid_solution:
            if solution.solution_shoulder_right.solution_wrist_up.is_valid_solution:
                if solution.solution_shoulder_right.solution_wrist_up.solution_elbow_up.is_valid_solution:
                    print(
                        f"Shoulder right, wrist up, elbow up: {solution.solution_shoulder_right.solution_wrist_up.solution_elbow_up.thetas}")
                if solution.solution_shoulder_right.solution_wrist_up.solution_elbow_down.is_valid_solution:
                    print(
                        f"Shoulder right, wrist up, elbow down: {solution.solution_shoulder_right.solution_wrist_up.solution_elbow_up.thetas}")
            if solution.solution_shoulder_right.solution_wrist_down.is_valid_solution:
                if solution.solution_shoulder_right.solution_wrist_down.solution_elbow_up.is_valid_solution:
                    print(
                        f"Shoulder right, wrist down, elbow up: {solution.solution_shoulder_right.solution_wrist_down.solution_elbow_up.thetas}")
                if solution.solution_shoulder_right.solution_wrist_down.solution_elbow_down.is_valid_solution:
                    print(
                        f"Shoulder right, wrist down, elbow down: {solution.solution_shoulder_right.solution_wrist_down.solution_elbow_up.thetas}")

    def compute_joint_angles(self, T06, debug=False):
        solution = InverseKinematicsSolution()

        #Theta 1
        P05 = np.dot(T06, [0, 0, -self.joint6_dh.d, 1])
        phi_1 = math.atan2(P05[1], P05[0])
        phi_2_cos = self.joint4_dh.d / math.sqrt(P05[0]**2 + P05[1]**2)
        phi_2 = math.acos(phi_2_cos)
        theta_1_1 = phi_1 + phi_2 + (np.pi / 2)
        theta_1_2 = phi_1 - phi_2 + (np.pi / 2)

        if debug:
            print(f"Theta 1: {theta_1_1:.3f}, {theta_1_2:.3f}")

        if not math.isnan(theta_1_1):
            solution.solution_shoulder_left = self.__compute_solution_for_theta_1(T06, theta_1_1, debug)
        else:
            solution.solution_shoulder_left = InverseKinematicsWristSolution().is_valid_solution = False
        if not math.isnan(theta_1_2):
            solution.solution_shoulder_right = self.__compute_solution_for_theta_1(T06, theta_1_2, debug)
        else:
            solution.solution_shoulder_right = InverseKinematicsWristSolution().is_valid_solution = False

        if debug:
            self.__print_all_solutions(solution)

        return solution

    def get_solution_for_config_id(self, solution, config_id):
        if config_id == 0:
            return solution.solution_shoulder_left.solution_wrist_up.solution_elbow_up.thetas
        elif config_id == 1:
            return solution.solution_shoulder_left.solution_wrist_up.solution_elbow_down.thetas
        elif config_id == 2:
            return solution.solution_shoulder_left.solution_wrist_down.solution_elbow_up.thetas
        elif config_id == 3:
            return solution.solution_shoulder_left.solution_wrist_down.solution_elbow_down.thetas
        elif config_id == 4:
            return solution.solution_shoulder_right.solution_wrist_up.solution_elbow_up.thetas
        elif config_id == 5:
            return solution.solution_shoulder_right.solution_wrist_up.solution_elbow_down.thetas
        elif config_id == 6:
            return solution.solution_shoulder_right.solution_wrist_down.solution_elbow_up.thetas
        elif config_id == 7:
            return solution.solution_shoulder_right.solution_wrist_down.solution_elbow_down.thetas
        else:
            raise Exception("invalid config solution id")

    def get_best_solution_for_config_id(self, T06, config_id):
        solution = self.compute_joint_angles(T06)
        if self.is_valid_solution_by_config_id(solution, config_id):
            return self.get_solution_for_config_id(solution, config_id)
        else:
            index = config_id + 1
            checked_all = False
            while not checked_all:
                if index >= 8:
                    index = 0
                if index == config_id:
                    print('Found no valid solutions..')
                    return None
                if self.is_valid_solution_by_config_id(solution, index):
                    return self.get_solution_for_config_id(solution, index)
                index += 1

    def is_valid_solution_by_config_id(self, solution, config_id):
        if 0 <= config_id < 4 and solution.solution_shoulder_left.is_valid_solution:
            if 0 <= config_id < 2 and solution.solution_shoulder_left.solution_wrist_up.is_valid_solution:
                if config_id == 0 and solution.solution_shoulder_left.solution_wrist_up.solution_elbow_up.is_valid_solution:
                    return True
                if config_id == 1 and solution.solution_shoulder_left.solution_wrist_up.solution_elbow_down.is_valid_solution:
                    return True
            if 2 <= config_id < 4 and solution.solution_shoulder_left.solution_wrist_down.is_valid_solution:
                if config_id == 2 and solution.solution_shoulder_left.solution_wrist_down.solution_elbow_up.is_valid_solution:
                    return True
                if config_id == 3 and solution.solution_shoulder_left.solution_wrist_down.solution_elbow_down:
                    return True
        if 4 <= config_id < 8 and solution.solution_shoulder_right.is_valid_solution:
            if 4 <= config_id < 6 and solution.solution_shoulder_right.solution_wrist_up.is_valid_solution:
                if config_id == 4 and solution.solution_shoulder_right.solution_wrist_up.solution_elbow_up.is_valid_solution:
                    return True
                if config_id == 5 and solution.solution_shoulder_right.solution_wrist_up.solution_elbow_down.is_valid_solution:
                    return True
            if 6 <= config_id < 8 and solution.solution_shoulder_right.solution_wrist_down.is_valid_solution:
                if config_id == 6 and solution.solution_shoulder_right.solution_wrist_down.solution_elbow_up.is_valid_solution:
                    return True
                if config_id == 7 and solution.solution_shoulder_right.solution_wrist_down.solution_elbow_down.is_valid_solution:
                    return True
        else:
            return False


    def get_current_configuration_id(self, joint_angles):
        T06 = self.forward_kinematics.compute_0_to_6_matrix(joint_angles)
        solution = self.compute_joint_angles(T06)
        differences = np.full(8, 1000)
        if solution.solution_shoulder_left.is_valid_solution:
            if solution.solution_shoulder_left.solution_wrist_up.is_valid_solution:
                if solution.solution_shoulder_left.solution_wrist_up.solution_elbow_up.is_valid_solution:
                    differences[0] = 0
                if solution.solution_shoulder_left.solution_wrist_up.solution_elbow_down.is_valid_solution:
                    differences[1] = 0
            if solution.solution_shoulder_left.solution_wrist_down.is_valid_solution:
                if solution.solution_shoulder_left.solution_wrist_down.solution_elbow_up.is_valid_solution:
                    differences[2] = 0
                if solution.solution_shoulder_left.solution_wrist_down.solution_elbow_down:
                    differences[3] = 0
        if solution.solution_shoulder_right.is_valid_solution:
            if solution.solution_shoulder_right.solution_wrist_up.is_valid_solution:
                if solution.solution_shoulder_right.solution_wrist_up.solution_elbow_up.is_valid_solution:
                    differences[4] = 0
                if solution.solution_shoulder_right.solution_wrist_up.solution_elbow_down.is_valid_solution:
                    differences[5] = 0
            if solution.solution_shoulder_right.solution_wrist_down.is_valid_solution:
                if solution.solution_shoulder_right.solution_wrist_down.solution_elbow_up.is_valid_solution:
                    differences[6] = 0
                if solution.solution_shoulder_right.solution_wrist_down.solution_elbow_down.is_valid_solution:
                    differences[7] = 0
        for i in range(6):
            if solution.solution_shoulder_left.is_valid_solution:
                if solution.solution_shoulder_left.solution_wrist_up.is_valid_solution:
                    if solution.solution_shoulder_left.solution_wrist_up.solution_elbow_up.is_valid_solution:
                        differences[0] += abs(joint_angles[i] - solution.solution_shoulder_left.solution_wrist_up.solution_elbow_up.thetas[i])
                    if solution.solution_shoulder_left.solution_wrist_up.solution_elbow_down.is_valid_solution:
                        differences[1] += abs(joint_angles[i] - solution.solution_shoulder_left.solution_wrist_up.solution_elbow_down.thetas[i])
                if solution.solution_shoulder_left.solution_wrist_down.is_valid_solution:
                    if solution.solution_shoulder_left.solution_wrist_down.solution_elbow_up.is_valid_solution:
                        differences[2] += abs(joint_angles[i] - solution.solution_shoulder_left.solution_wrist_down.solution_elbow_up.thetas[i])
                    if solution.solution_shoulder_left.solution_wrist_down.solution_elbow_down:
                        differences[3] += abs(joint_angles[i] - solution.solution_shoulder_left.solution_wrist_down.solution_elbow_down.thetas[i])
            if solution.solution_shoulder_right.is_valid_solution:
                if solution.solution_shoulder_right.solution_wrist_up.is_valid_solution:
                    if solution.solution_shoulder_right.solution_wrist_up.solution_elbow_up.is_valid_solution:
                        differences[4] += abs(joint_angles[i] - solution.solution_shoulder_right.solution_wrist_up.solution_elbow_up.thetas[i])
                    if solution.solution_shoulder_right.solution_wrist_up.solution_elbow_down.is_valid_solution:
                        differences[5] += abs(joint_angles[i] - solution.solution_shoulder_right.solution_wrist_up.solution_elbow_down.thetas[i])
                if solution.solution_shoulder_right.solution_wrist_down.is_valid_solution:
                    if solution.solution_shoulder_right.solution_wrist_down.solution_elbow_up.is_valid_solution:
                        differences[6] += abs(joint_angles[i] - solution.solution_shoulder_right.solution_wrist_down.solution_elbow_up.thetas[i])
                    if solution.solution_shoulder_right.solution_wrist_down.solution_elbow_down.is_valid_solution:
                        differences[7] += abs(joint_angles[i] - solution.solution_shoulder_right.solution_wrist_down.solution_elbow_down.thetas[i])
        print(differences)
        return np.argmin(differences)
