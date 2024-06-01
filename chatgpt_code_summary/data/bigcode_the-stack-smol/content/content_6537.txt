"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
part of mavPySim 
    - Beard & McLain, PUP, 2012
    - Update history:  
        12/20/2018 - RWB
        2/24/2020
"""
import sys
sys.path.append('..')
import numpy as np

# load message types
from message_types.msg_state import msgState

import parameters.aerosonde_parameters as MAV
from tools.rotations import Quaternion2Rotation, Quaternion2Euler, skew, quat_prod
import mavsim_python_chap5_model_coef as chap5

class mavDynamics:
    def __init__(self, Ts):
        self._ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        # We will also need a variety of other elements that are functions of the _state and the wind.
        # self.true_state is a 19x1 vector that is estimated and used by the autopilot to control the aircraft:
        # true_state = [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        self._state = np.array([[MAV.pn0],  # (0)
                                [MAV.pe0],   # (1)
                                [MAV.pd0],   # (2)
                                [MAV.u0],    # (3)
                                [MAV.v0],    # (4)
                                [MAV.w0],    # (5)
                                [MAV.e0],    # (6)
                                [MAV.e1],    # (7)
                                [MAV.e2],    # (8)
                                [MAV.e3],    # (9)
                                [MAV.p0],    # (10)
                                [MAV.q0],    # (11)
                                [MAV.r0]])   # (12)
        
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        self._update_velocity_data()
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[], [], []])
        
        ur = self._state.item(3)
        vr = self._state.item(4)
        wr = self._state.item(5)
        
        self._Va = np.sqrt(ur**2 + vr**2 + wr**2)
        self._alpha = np.arctan2(wr,ur)
        self._beta = np.arcsin(vr/self._Va)
        
        # initialize true_state message
        self.true_state = msgState()

    ###################################
    # public functions
    def update(self, delta, wind):
        """
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        """
        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments(delta)

        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._ts_simulation
        k1 = self._derivatives(self._state, forces_moments)
        k2 = self._derivatives(self._state + time_step/2.*k1, forces_moments)
        k3 = self._derivatives(self._state + time_step/2.*k2, forces_moments)
        k4 = self._derivatives(self._state + time_step*k3, forces_moments)
        self._state += time_step/6 * (k1 + 2*k2 + 2*k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = np.sqrt(e0**2+e1**2+e2**2+e3**2)
        self._state[6][0] = self._state.item(6)/normE
        self._state[7][0] = self._state.item(7)/normE
        self._state[8][0] = self._state.item(8)/normE
        self._state[9][0] = self._state.item(9)/normE

        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)

        # update the message class for the true state
        self._update_true_state()

    def external_set_state(self, new_state):
        self._state = new_state

    ###################################
    # private functions
    def _derivatives(self, x, u):
        """
        for the dynamics xdot = f(x, u), returns fdot(x, u)
        """
        # Get force, moment (torque)
        f_b = u[:3]
        m_b = u[3:]
        # Get position, velocity, quaternion (rotation), angular velocity 
        r_i = x[:3] # wrt to i-frame
        v_b = x[3:6] # wrt to i-frame
        q_ib = x[6:10] # for rotation b to i-frame
        w_b = x[10:] # wrt to b-frame
        
        # Normalize quat. -> rotation
        q_ib = q_ib/np.linalg.norm(q_ib) # normalize
        R_ib = Quaternion2Rotation(q_ib)
        
        # Compute equations of motion
        # d/dt(r_i) 
        rdot_i = R_ib @ v_b
        
        # d/dt(v_b)
        vdot_b = (1/MAV.mass)*f_b-skew(w_b) @ v_b
        
        # d/dt(q_ib)
        wq_ib = np.zeros((4,1))
        wq_ib[1:] = w_b
        qdot_ib = 0.5 * quat_prod(wq_ib, q_ib)
        wt_b = skew(w_b)
        
        # d/dt(w_b)
        wdot_b = np.linalg.inv(MAV.J) @ (m_b - (wt_b @ (MAV.J @ w_b)))
        
        x_out = np.concatenate([rdot_i,vdot_b,qdot_ib,np.array(wdot_b)],axis = 0)
        return x_out

    def _update_velocity_data(self, wind=np.zeros((6,1))):
        steady_state = wind[0:3]
        gust = wind[3:6]
        ur = self._state.item(3) - steady_state[0] - gust[0]
        vr = self._state.item(4) - steady_state[1] - gust[1]
        wr = self._state.item(5) - steady_state[2] - gust[2]
        
        self._Va = np.sqrt(ur**2 + vr**2 + wr**2)
        self._alpha = np.arctan2(wr,ur)
        self._beta = np.arcsin(vr/self._Va)


    def thrust_from_prop(self, delta_t):
        # compute thrust and torque due to propeller (See addendum by McLain)
        # map delta_t throttle command (0 to 1) into motor input voltage
            
        V_in = MAV.V_max * delta_t
        KQ = MAV.KQ
        
        # Quadratic formula to solve for motor speed
        a = MAV.C_Q0 * MAV.rho * np.power(MAV.D_prop, 5) / ((2. * np.pi )**2 )
        b = (MAV.C_Q1 * MAV.rho * np.power(MAV.D_prop, 4) / (2.*np.pi)) * self._Va + KQ**2/MAV.R_motor
        c = MAV.C_Q2 * MAV.rho * np.power(MAV.D_prop, 3) * self._Va**2 - (KQ / MAV.R_motor ) * V_in + KQ * MAV.i0
        # Consider only positive root
        Omega_op = (-b + np.sqrt(b**2 - 4*a* c)) / (2. * a )
        # compute advance ratio
        J_op = 2 * np.pi * self._Va / (Omega_op * MAV.D_prop)
        # compute nondimensionalized coefficients of thrust and torque
        C_T = MAV.C_T2 * J_op **2 + MAV.C_T1 * J_op + MAV.C_T0
        C_Q = MAV.C_Q2 * J_op **2 + MAV.C_Q1 * J_op + MAV.C_Q0
        # add thrust and torque due to propeller
        n = Omega_op / (2 * np.pi )
        fx = MAV.rho * n**2 * np.power(MAV.D_prop, 4) * C_T
        Mx = -MAV.rho * n**2 * np.power(MAV.D_prop, 5) * C_Q
        return fx,Mx
    
    def sigma(self,alpha):
        # pseudo sigmoid functions with cutoff +- alpha_0, returns coef btw 0 and 1
        a1 = -MAV.M * (alpha - MAV.alpha0)
        a2 = MAV.M * (alpha + MAV.alpha0)
        sigma_alpha = (1 + np.exp(a1)+np.exp(a2)) / ((1+np.exp(a1))*(1+np.exp(a2)))
        return sigma_alpha
    
    def CL(self,alpha):
        CL0 = MAV.C_L_0
        CLA = MAV.C_L_alpha
        sigma_alpha = self.sigma(alpha)
        # returns lift coefficient using eq 4.9
        CL_alpha = (1-sigma_alpha)*(CL0 + CLA*alpha) + sigma_alpha*(2*np.sign(alpha)*np.sin(alpha)**2 * np.cos(alpha))
        return CL_alpha
    
    def CD(self,alpha):
        # returns drag coefficient using eq 4.11
        CD_alpha = MAV.C_D_p + (MAV.C_L_0 + MAV.C_L_alpha*alpha)**2/(np.pi*MAV.e*MAV.AR)
        return CD_alpha
    
    def Cx(self,alpha):
        return -self.CD(alpha)*np.cos(alpha) + self.CL(alpha)*np.sin(alpha)
    
    def Cx_q(self,alpha):
        return -MAV.C_D_q*np.cos(alpha) + MAV.C_L_q*np.sin(alpha)
    
    def Cx_deltae(self,alpha):
        return -MAV.C_D_delta_e*np.cos(alpha) + MAV.C_L_delta_e*np.sin(alpha)
    
    def Cz(self,alpha):
        return -self.CD(alpha)*np.sin(alpha)-self.CL(alpha)*np.cos(alpha)
    
    def Cz_q(self,alpha):
        return -MAV.C_D_q*np.sin(alpha)-MAV.C_L_q*np.cos(alpha)
    
    def Cz_deltae(self,alpha):
        return -MAV.C_D_delta_e*np.sin(alpha)-MAV.C_L_delta_e*np.cos(alpha)
    
    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_e, delta_a, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        p = self._state.item(10)
        q = self._state.item(11)
        r = self._state.item(12)

        delta_e = delta.item(0)
        delta_a = delta.item(1)
        delta_r = delta.item(2)
        delta_t = delta.item(3)
        
        # Gravitational Components of Force, Moments = 0
        mg = MAV.mass*MAV.gravity
        fx_grav = -mg*np.sin(theta)
        fy_grav = mg* np.cos(theta) * np.sin(phi)
        fz_grav = mg* np.cos(theta) * np.cos(phi)
                    
        # Thrust Components of Force and Moments
        fx_thrust,Mx_thrust = self.thrust_from_prop(delta_t)
        fy_thrust = 0
        fz_thrust = 0
        My_thrust = 0
        Mz_thrust = 0
        
        # Aerodynamic Components of Forces and Moments
        b = MAV.b
        cyp = MAV.C_Y_p
        cyr = MAV.C_Y_r
        cydeltaa = MAV.C_Y_delta_a
        cydeltar = MAV.C_Y_delta_r
                
        aero_coef = 0.5*MAV.rho*self._Va**2*MAV.S_wing
        fx_aero = aero_coef * (self.Cx(self._alpha) + self.Cx_q(self._alpha)*MAV.c/(2*self._Va)*q + self.Cx_deltae(self._alpha)*delta_e)
        fy_aero = aero_coef * (MAV.C_Y_0 + MAV.C_Y_beta*self._beta + MAV.C_Y_p*b/(2*self._Va)*p + cyr * b/(2*self._Va)*r + cydeltaa * delta_a + cydeltar* delta_r)
        fz_aero = aero_coef * (self.Cz(self._alpha) + self.Cz_q(self._alpha)*MAV.c/(2*self._Va)*q + self.Cz_deltae(self._alpha)*delta_e)
        Mx_aero = aero_coef * MAV.b * (MAV.C_ell_0 + MAV.C_ell_beta*self._beta + MAV.C_ell_p*b/(2*self._Va)*p + MAV.C_ell_r*b/(2*self._Va)*r + MAV.C_ell_delta_a*delta_a + MAV.C_ell_delta_r*delta_r)
        My_aero = aero_coef * MAV.c * (MAV.C_m_0 + MAV.C_m_alpha*self._alpha + MAV.C_m_q*MAV.c/(2*self._Va)*q + MAV.C_m_delta_e*delta_e)
        Mz_aero = aero_coef * MAV.b * (MAV.C_n_0 + MAV.C_n_beta*self._beta + MAV.C_n_p*MAV.b/(2*self._Va)*p + MAV.C_n_r*MAV.b/(2*self._Va)*r + MAV.C_n_delta_a*delta_a + MAV.C_n_delta_r*delta_r)
        
        
        
        fx = fx_grav + fx_aero + fx_thrust
        fy = fy_grav + fy_aero + fy_thrust
        fz = fz_grav + fz_aero + fz_thrust
        # print('fx = ',fx)
        # print('fy = ',fy)
        # print('fz = ',fz)
        Mx = Mx_aero + Mx_thrust
        My = My_aero + My_thrust
        Mz = Mz_aero + Mz_thrust
        # print('Mx = ',Mx)
        # print('My = ',My)
        # print('Mz = ',Mz)

        self._forces[0] = fx
        self._forces[1] = fy
        self._forces[2] = fz
        fm = np.reshape(np.array([fx, fy, fz, Mx, My, Mz]),[6,1])
        return fm

    def _update_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        pdot = Quaternion2Rotation(self._state[6:10]) @ self._state[3:6]
        self.true_state.pn = self._state.item(0)
        self.true_state.pe = self._state.item(1)
        self.true_state.h = -self._state.item(2)
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = np.linalg.norm(pdot)
        self.true_state.gamma = np.arcsin(pdot.item(2) / self.true_state.Vg)
        self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = self._wind.item(0)
        self.true_state.we = self._wind.item(1)
        