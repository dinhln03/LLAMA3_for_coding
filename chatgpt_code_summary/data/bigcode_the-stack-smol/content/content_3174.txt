# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 22:59:51 2019

@author: Sravan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 22:36:21 2019

@author: Sravan
"""
import csv
import numpy as np
from scipy.spatial.distance import pdist, squareform, euclidean, cdist

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import scipy.integrate as integrate
import matplotlib.animation as animation


"""
Variables: Wind speed, Air traffic (# of drones), Obstacles (Trees, Buildings) 
Fixed: Distance, Air Resistance, Gravity, Battery level
Rules: Drone Speed (Air traffic, Wind speed, Battery level), Collisions (Drone position)
Study: Time, Speed
Movement: v_air = sqrt(mg/(nAρ)), p = 1.22 kg m^-3, A = 1 m^2
½cρAv2 = mgtanθ, c = drag coefficient
P = ½ρnAv_air(v_air2 – v2sin2θ)
Collisions: Drone - Increase/Decrease Speed, 2) Change path- increasing elevation

https://www.research-drone.com/en/extreme_climb_rate.html
https://en.wikipedia.org/wiki/Amazon_Prime_Air
https://homepages.abdn.ac.uk/nph120/meteo/DroneFlight.pdf
"""
class ParticleBox:
    """Orbits class
    
    init_state is an [N x 6] array, where N is the number of particles:
       [[xi1, yi1, zi1, xf1, yf1, zf1, vx1, vy1, vz1, t1],
        [xi2, yi2, zi2, xf2, yf2, zf2, vx2, vy2, vz2, t2],
        ...               ]

    bounds is the size of the box: [xmin, xmax, ymin, ymax, zmin, zmax]
    """
    def __init__(self,
                 drones = 1,
                 wind = [0, 0, 0],
                 obstacles = 0,
                 bounds = [-32000, 32000, -32000, 32000, 0, 150],
                 size = 1.5,
                 max_height = 122,
                 max_speed = 22.34,
                 acc = 7,
                 M = 25.0,
                 G = 9.81):
        self.drones = drones
        self.wind = wind
        self.size = size
        self.G = G
        self.max_height = max_height
        self.max_speed = max_speed
        self.acc_vert = acc
        self.acc_vert_eff = acc + G
        self.acc_hor = acc
        self.obstacles = 0
        self.obstacles_size = 40
        self.time_elapsed = 0
        self.bounds = bounds
        
        np.random.seed(0)
        init_state = np.random.random((drones, 10))
        init_state[:, :2] -= 0.5
        init_state[:, :2] *= bounds[1]*2
        init_state[:, 2:] = 0.0
        for i in range(len(init_state)):
            vecs = [64000.0, 64000.0]
            while vecs[0] > bounds[1] or vecs[0] < bounds[0] or vecs[1] > bounds[3] or vecs[1] < bounds[2]:
                vecs = np.random.standard_normal(2)
                mags = np.linalg.norm(vecs)
                vecs /= mags
                vecs *= 16000
                vecs += init_state[i, :2]
            init_state[i, 3:5] =vecs
        
        if obstacles > 0:
            np.random.seed(1)
            obs_state = np.random.random((obstacles, 3))
            obs_state[:, :3] -= 0.5
            obs_state[:, :2] *= bounds[1]*2
            obs_state[:, 2] *= bounds[5]*2
        
        self.init_state = np.asarray(init_state, dtype=float)
        #self.obs_state = np.asarray(obs_state, dtype=float)
        self.M = M * np.ones(self.init_state.shape[0])
        self.state = self.init_state.copy()
        
        #update velocity
        self.state[:, 6] = self.wind[0]
        self.state[:, 7] = self.wind[1]
        self.state[:, 8] = self.wind[2]

    def step(self, dt):
        """step once by dt seconds"""
        self.time_elapsed += dt
        
        # find distance to goal
        D = cdist(self.state[:, :3], self.state[:, 3:6], 'euclidean')
        ind, din = np.where(D > 122)
        uniqua = (ind == din)
        ind = ind[uniqua]
        
        # update velocities of individual drones
        for i in zip(ind):
            #velocity vector
            v = self.state[i, 8]
            v_avg = v
            a_ver = self.acc_vert
            a_ver_eff = self.acc_vert_eff
            height = self.max_height - self.state[i, 2]
            print(height)
            if height > 0:
                n = 1
                if v > 0:
                    n = v / abs(v)
                stop = n * v**2/(2 * a_ver)
                t_end = abs(v / a_ver)
                
                b1 = (v**2 + t_end**2)**(0.5)
                b2 = ((v + n * a_ver * dt)**2 + (t_end + dt)**2)**(0.5)
                s1 = ((a_ver * dt)**2 + dt**2)**(0.5)
                s2 = dt * 2
                P = (b2 - b1) + s1 + s2
                t = ((P/2) * (P/2 - s1) * (P/2 - s2) * (P/2 - b2 + b1))**(0.5)
                h = 2 * t / (b2 - b1)
                area = n * (t + (b2 - b1) * h)
                
                if (t_end <= dt and stop > (height - area)):
                    v_avg = 0
                    self.state[i, 8] = 0
                    self.state[i, 2] = self.max_height
                elif (stop > (height - area)):
                    t_max = 0
                    if stop < height:
                        a = 2 * (a_ver)**2
                        b = 4 * (a_ver) * v
                        c = v**2 - 2 * a_ver * height
                        t_max = (-b + (b**2 - 4 * a * c)**(0.5)) / (2 * a)
                    v_max = v + a_ver * (t_max / dt)
                    v_end = 2 * v_max - v - a_ver * dt
                    v_avg = ((v_max + v) / 2) * (t_max / dt) + ((v_max + v_end) / 2) * ((dt - t_max) / dt)
                    self.state[i, 8] = v_end
                else:
                    v_avg = v + a_ver * dt / 2
                    self.state[i, 8] += a_ver * dt
            elif height < 0:
                n = v / abs(v)
                stop = n * v**2/(2 * a_ver_eff)
                t_end = abs(v / a_ver_eff)
                
                b1 = (v**2 + t_end**2)**(0.5)
                b2 = ((v + n * a_ver_eff * dt)**2 + (t_end + dt)**2)**(0.5)
                s1 = ((a_ver_eff * dt)**2 + dt**2)**(0.5)
                s2 = dt * 2
                P = (b2 - b1) + s1 + s2
                t = ((P/2) * (P/2 - s1) * (P/2 - s2) * (P/2 - b2 + b1))**(0.5)
                h = 2 * t / (b2 - b1)
                area = n * (t + (b2 - b1) * h)
                
                if (t_end <= dt and abs(stop) <= abs(height)):
                    v_avg = (v / 2) * (t_end / dt)
                    self.state[i, 8] = v + a_ver_eff * t_end
                elif (stop < (height - area)):
                    v_max = (height * (2 * a_ver_eff))**(0.5)
                    t_max = (v_max - v)/a_ver_eff
                    v_end = 2 * v_max - v - a_ver_eff * dt
                    v_avg = ((v_max + v) / 2) * (t_max / dt) + ((v_max + v_end) / 2) * ((dt - t_max) / dt)
                    self.state[i, 8] = v_end
                else:
                    v_avg = v - a_ver_eff * dt / 2
                    self.state[i, 8] = v - a_ver_eff * dt
            else:
                self.state[i, 8] += 0 * dt
                
            
            self.state[i, 2] += v_avg * dt
            
            # unit vector
            r = self.state[i, 3:5] - self.state[i, :2]
            m = np.linalg.norm(r)
            u = r / m
            
            #accelearting horizontal
            a_hor = self.acc_hor
            v_hor = self.state[i, 6:8]
            h = np.linalg.norm(v_hor)
            
            stop = h**2/(2 * a_hor)
            t_end = h / a_hor
            
            b1 = (h**2 + t_end**2)**(0.5)
            b2 = ((h + a_hor * dt)**2 + (t_end + dt)**2)**(0.5)
            s1 = ((a_hor * dt)**2 + dt**2)**(0.5)
            s2 = dt*2
            P = (b2 - b1) + s1 + s2
            t = ((P/2) * (P/2 - s1) * (P/2 - s2) * (P/2 - b2 + b1))**(0.5)
            s = 2 * t / (b2 - b1)
            area = (t + (b2 - b1) * s)
            
            if (t_end <= dt and stop < area):
                v_hor = (h / 2) * (t_end / dt)
                self.state[i, 6:8] = (h - (a_hor * t_end)) * u
            elif (stop > (m - area)):
                v_max = (m * (2 * a_hor))**(0.5)
                t_max = (v_max - h)/a_hor
                v_end = 2 * v_max - h - a_hor * dt
                v_hor = ((v_max + h) / 2) * (t_max / dt) + ((v_max + v_end) / 2) * ((dt - t_max) / dt)
                self.state[i, 6:8] = v_end * u
            else:
                v_hor = h + a_hor * dt / 2
                self.state[i, 6:8] = (h + a_hor * dt) * u
            
            self.state[i, :2] += (v_hor * dt) * u

        #find drones hovering
        done, fund = np.where(D <= 122)
        uniquo = (done == fund)
        done = done[uniquo]
        for d in zip(done):
            print("here")
            #velocity vector
            v = self.state[i, 8]
            v_avg = v
            a_ver_eff = self.acc_vert_eff
            
            #accelerating negative z
            n = -1
            if v < 0:
                n = v / abs(v)
            stop = n * v**2/(2 * a_ver_eff)
            t_end = abs(v / a_ver_eff)
            
            b1 = (v**2 + t_end**2)**(0.5)
            b2 = ((v + n * a_ver_eff * dt)**2 + (t_end + dt)**2)**(0.5)
            s1 = ((a_ver_eff * dt)**2 + dt**2)**(0.5)
            s2 = dt * 2
            P = (b2 - b1) + s1 + s2
            t = ((P/2) * (P/2 - s1) * (P/2 - s2) * (P/2 - b2 + b1))**(0.5)
            h = 2 * t / (b2 - b1)
            area = n * (t + (b2 - b1) * h)
            
            if (t_end <= dt and stop > area):
                v_avg = (v / 2) * (t_end / dt)
                self.state[i, 8] = v + a_ver_eff * t_end
                self.state[i, 9] = self.time_elapsed
            elif (stop < (-self.state[i, 2] - area)):
                v_max = ((-self.state[i, 2]) * (2 * a_ver_eff))**(0.5)
                t_max = (v_max - v)/a_ver_eff
                v_end = 2 * v_max - v - a_ver_eff * dt
                v_avg = ((v_max + v) / 2) * (t_max / dt) + ((v_max + v_end) / 2) * ((dt - t_max) / dt)
                self.state[i, 8] = v_end
            else:
                v_avg = v - a_ver_eff * dt / 2
                self.state[i, 8] = v - a_ver_eff * dt
            
            self.state[i, 2] += v_avg * dt


        E = squareform(pdist(self.state[:, :3], 'euclidean'))
        ind1, ind2 = np.where(E < (2 * self.size))
        unique = (ind1 < ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]
        
        for i1, i2 in zip(ind1, ind2):
            if (self.state[i1, 2] > self.state[i2, 2]):
                self.state[i1, 8] += (self.acc_vert) * dt
                self.state[i2, 8] -= (self.acc_vert_eff) * dt
            else:
                self.state[i1, 8] -= (self.acc_vert) * dt
                self.state[i2, 8] += (self.acc_vert_eff) * dt
                    
        if self.obstacles > 0:
            DO = np.vstack([self.state[:, :3].copy(), self.obs_state.copy()])
            F = squareform(pdist(DO, 'euclidean'))
            d_rone, obs = np.where(F < (2 * self.obstacles_size))
            unique = (d_rone < obs and obs >= self.drones)
            d_rone = d_rone[unique]
            obs = obs[unique]
            
            for d, o in zip(d_rone, obs):
                if (self.obs_state[o-self.drones, 2] < 110 and self.state[d, 2] < self.obs_state[o-self.drones, 2]):
                    self.state[d, 8] += self.acc_vert * dt
                else:
                    r = self.state[d, 3:5] - self.state[d, :2]
                    ro = self.obs_state[o-self.drones, :2] - self.state[d, :2]
                    
                    r_rel = np.cross(r, ro)
                    if (r_rel[2] > 0):
                        self.state[d, 6] += self.acc_hor * dt
                        self.state[d, 7] += self.acc_hor * dt
                    else:
                        self.state[d, 6] -= self.acc_hor * dt
                        self.state[d, 7] -= self.acc_hor * dt
    
        #restrict velocity
        np.clip(self.state[:, 6], -self.max_speed + self.wind[0], self.max_speed + self.wind[0])
        np.clip(self.state[:, 7], -self.max_speed + self.wind[1], self.max_speed + self.wind[1])


#------------------------------------------------------------
# set up initial state

box = ParticleBox()
dt = 1. # 1 fps
    
#ani = animation.FuncAnimation(fig, animate, frames=600, interval=10, init_func=init)
for i in range(10):
    box.step(dt)


#final = np.hstack([box.init_state[:, :3], box.state[:, 3:]])

#with open('people.csv', 'w') as writeFile:
#    writer = csv.writer(writeFile)
#    writer.writerows(final) #2d list

"""with open('initial.csv', 'w') as writeInit:
    writer = csv.writer(writeInit)
    writer.writerows(box.init_state)
    
writeInit.close()
    """
    
with open('final_2.csv', 'w') as writeFin:
    writer = csv.writer(writeFin)
    writer.writerows(box.init_state)
    writer.writerows(box.state)

writeFin.close()

print(box.state)