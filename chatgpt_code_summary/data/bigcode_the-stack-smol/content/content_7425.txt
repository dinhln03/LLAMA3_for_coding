# Code made for Sergio Andrés Díaz Ariza
# 23 March 2021
# License MIT
# Transport Phenomena: Python Program-Assesment 3.3

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()


class Wire_Param:

    def Power_from_volt(self, V, R_Km, Dis_insul, Diameter_mm):
        As = (np.pi * (Diameter_mm / 1000) ** 2) / 4
        R = R_Km * Dis_insul * As / 1000
        W = (V ** 2) / R  # Power generated = Egen
        return W  # [w/m^3]

    def Param_m(self, h, Kcopper, Diameter_mm):
        m = np.sqrt((4 * h * 0.2) / (Kcopper * (Diameter_mm / 1000)))
        return m


class Coeff_vector_B:

    def Eq_1(self, h, Tsurr, Egen, K, L):
        b1 = h * Tsurr - (Egen * (-L / 2)) + (h * (Egen * ((-L / 2) ** 2)) / 2 * K)
        return b1

    def Eq_2(self, Tsurr):
        b2 = Tsurr
        return b2

    def Eq_3(self):
        b3 = 0
        return b3

    def Eq_4(self):
        b4 = 0
        return b4


class T_profile:

    def T_z_to_L_2(self, Egen, z1, Kcopper, C1, C2):
        T = ((-Egen * ((z1) ** 2)) / 2 * Kcopper) + C1 * (z1) + C2
        return T

    def T_z_from_L_2_to_L(self, Tsurr, C3, C4, Param_m, z2):
        T = Tsurr + (C3 * np.exp((Param_m * z2))) + (C4 * np.exp((-Param_m * z2)))
        return T


Wire = Wire_Param()
Vect_B = Coeff_vector_B()
Profile = T_profile()

Egen = Wire.Power_from_volt(30e-6, 5.32, 0.2, 2.32)  # [W/m^3]
Param_m = Wire.Param_m(15, 386, 2.32)  # Value L--> must be positive fo calculus

b1 = Vect_B.Eq_1(15, 25 + 273.15, Egen, 386, -0.2)
b2 = Vect_B.Eq_2(25 + 273.15)
b3 = Vect_B.Eq_3()
b4 = Vect_B.Eq_4()

B = np.array([b1, b2, b3, b4])
A = np.array([[-386 + (15 * (-0.2)), 15, 0, 0], [0, 1, -1, -1], [1, 0, -Param_m, Param_m],
              [0, 0, (np.exp(Param_m * 0.2)) * ((-386 * Param_m) - 15),
               (np.exp(-Param_m * 0.2)) * ((386 * Param_m) - 15)]])

# A_inv = np.linalg.inv(A) ---> We can find the solution directly
C = np.linalg.solve(A, B)

# solve for Temperature Profile
T_L_2 = Profile.T_z_to_L_2(Egen, np.arange(-0.2, 0.0, 1e-6), 386, C[0], C[1])
T_L = Profile.T_z_from_L_2_to_L(25 + 273.15, C[2], C[3], Param_m, np.arange(0.0, 0.2, 1e-6))

print(T_L_2)
print(T_L)
plt.figure(1)
plt.plot(np.arange(0, 0.2, 1e-6), T_L_2, label='$0<L<0.2$')
plt.plot(np.arange(0.2, 0.4, 1e-6), T_L, label='$0.2<L<0.4$')
plt.title("Temperature Profile in a Wie\n with Insulated part", fontsize=16)
plt.ylabel("Temperature $[K]$", fontsize=14)
plt.xlabel("Long of wire $[m]$ ", fontsize=14)
plt.legend()
plt.show()
