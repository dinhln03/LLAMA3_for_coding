# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:39:59 2020

@author: nicholls
"""

import os
import numpy as np
import matplotlib.pyplot as plt

#%%

class AngularSpreadCalc():
    """ class for calculating how angular spread changes with iterations:
        Inputs:
            iterations: maxinum number of iterations to calculate for (e.g. 500)
            acceptance angle:   acceptance angle of analyser
            energy:     initial energy of scattered electrons (eV)
    """

    def __init__(self, iterations, acceptance_angle, energy, width=None):
        self.iterations = iterations
        self.acceptance_angle = acceptance_angle
        self.energy = energy
        self.width = width

    def gen_lorentzian_cross_section(self):

        self.cross_section_x = np.arange(-90, 90, 1)
        y = [self._lorentzian_cross_section(x, self.width) for x in self.cross_section_x] 

        self.cross_section_y = y
        return self.cross_section_y

    def _lorentzian_cross_section(self, x, width):
        position = 0
        intensity = 1
        
        l = intensity * 1 / (1 + ((position-x)/(width/2))**2)
        
        return l

    def plot_cross_section(self):
        """ Plot the raw imported nist data """
        plt.plot(self.cross_section_x, self.cross_section_y)
        plt.title('Cross Section')
        plt.xlabel('Angle')
        plt.show()


    def load_nist_cross_section(self, filename):
        """ Load nist data file of differential elastic scattering profile.
        Input:
            filename: filename of csv data from nist database
        Returns:
            cross_section_y: given cross section in range -90 to 90 deg """

        filepath = (os.path.dirname(os.path.abspath(__file__)).partition('controller')[0]
                    + '\\data\\NIST cross sections\\' + filename)

        data = np.genfromtxt(filepath, skip_header=10, delimiter=',')
        self.cross_section_y = self._convert_nist_data(data)

        self.cross_section_x = np.arange(-90, 90, 1)
        return self.cross_section_y


    def plot_nist(self):
        """ Plot the raw imported nist data """
        plt.plot(self.cross_section_x, self.cross_section_y)
        plt.title('NIST Data')
        plt.xlabel('Angle')
        plt.show()


    def run_convolution(self):
        """ Run convolution between the nist cross section and a sine curve
        representing initial scattering distribution.
        Returns:
            centered_data:  angular distribution spread after each scattering
                event
        """
        # normalise cross section by area under curve
        self.cross_section_y_norm = self.cross_section_y / np.sum(self.cross_section_y)
        # generate initial distribution of electron scatter:
        self.emitted_elctn_y = self._gen_electron_dist()
        self.emitted_elctn_x = np.arange(-90, 90, 1)
        # run convolution
        convolved_data = self._convolution(self.cross_section_y_norm,
                                           self.emitted_elctn_y,
                                           self.iterations)
        # center data and remove excess data (i.e. outside -90 to 90 range)
        self.centered_data = self._centre_data(convolved_data)

        return self.centered_data

    def plot_convolution_results(self):
        """ Plot convolution result to show angular distribution spread after
        each scattering event."""
        # plotting selected iterations:
        for n in [0, 1, 2, 5, 10, 20, 50]:
            plt.plot(self.emitted_elctn_x, self.centered_data[n], label=str(n))
            plt.xticks([-90, -60, -30, 0, 30, 60, 90])
            plt.xlabel('theta (degrees)')
            plt.ylabel('Intensity (a.u.)')
            plt.title('Angular distribution per scattering event')
            plt.legend(title='No. of iterations', loc='center left',
                       bbox_to_anchor=(1, 0.5))
        #plt.savefig('Convolution.png', dpi=600, bbox_inches='tight')
        plt.show()

    def limit_by_acceptance_angle(self):
        """ Limit the data to the acceptance angle of the analyser """
        # to set acceptance angle
        self.angle_limited = self._limit_by_constant_angle(self.centered_data,
                                                           self.acceptance_angle)
        #return self.angle_limited

    def plot_angle_limited(self):
        """ Plot the convolution results only in the accepted angle range"""
        # to plot angle limited data
        for n in [0, 1, 2, 5, 10, 20, 50]:
            plt.plot(self.emitted_elctn_x, self.angle_limited[n], label=str(n))
            plt.xticks([-90, -60, -30, 0, 30, 60, 90])
            plt.xlabel('theta (degrees)')
            plt.ylabel('Intensity (a.u.)')
            plt.title('Intensity distribution after scattering event')
            plt.legend(title='No. of iterations', loc='center left', bbox_to_anchor=(1, 0.5))
        #plt.savefig('angle convolution.png', dpi=600, bbox_inches='tight')
        plt.show()

    def calc_area_under_curve(self):
        """ Calculate area under each curve within acceptance angle,
        represents intensity that the detector sees"""
        sin = np.absolute(np.sin(np.arange(-90, 90, 1) * np.pi / 180))
        angle_integrated = self.angle_limited * sin * np.pi
        self.area_sum = np.sum(angle_integrated, axis=1)
        self.area_sum = self.area_sum / self.area_sum[0]
        return self.area_sum

    def plot_area_under_curve(self):
        """ Plot area under curve per scattering event / iteration """
        plt.plot(self.area_sum)
        plt.title('area under curve \n '
                  '(Energy: ' + str(self.energy) + ', Acceptance Angle: ' +
                  str(self.acceptance_angle) + ')')
        plt.xlabel('No. of iterations')
        plt.ylabel('Intensity a.u.')
        plt.show()

    def calc_area_ratio(self):
        """ Calculate the change in area ratio between iteration n and n-1"""
        # Change in ratio
        self.area_ratio_list = self._area_ratio_change(self.area_sum)
        return self.area_ratio_list

    def plot_area_ratio(self):
        """ Plot the change in area ratio per iteration """
        # to plot
        plt.plot(self.area_ratio_list)
        plt.title('Intensity ratio change per iteration \n '
                  '(Energy: ' + str(self.energy) + ' eV, Acceptance Angle: '
                  + str(self.acceptance_angle) + ')')
        plt.xlabel('Iterations')
        plt.ylabel('Area Ratio between iterations')
        #plt.savefig('Ratio change per iteration.png', dpi=600)
        plt.show()

    def _convert_nist_data(self, dataset):
        data = [n for n in dataset[:, 1]]
        data.reverse()
        data.extend([n for n in dataset[:, 1]][1:])
        data = data[90:270]
        return data

    def _gen_electron_dist(self):
        # x values
        self.emitted_elctn_x = np.arange(-90, 90, 1)
        # calculate y by cosine distribution
        self.emitted_elctn_y = np.array([(np.cos(np.pi * i / 180))
                                         for i in self.emitted_elctn_x])
        # normalise by area under the curve
        self.emitted_elctn_y = self.emitted_elctn_y / np.sum(self.emitted_elctn_y)

        return self.emitted_elctn_y

    def _convolution(self, cross_section, scatter, n):
        # empty list to contain arrays of the scattered electrons
        scattered_events = []
        # add the first entry for unscattered:
        scattered_events.append(scatter)
        # convolution n number of times:
        for i in range(n):
            # convolve cross section with last scattered
            z = np.convolve(cross_section, scattered_events[i])
            # add scattered to list
            scattered_events.append(z)

        return scattered_events


    def _centre_data(self, scattered_data_list):

        data_cropped = []
        for indx, scattering_event in enumerate(scattered_data_list):

            centre = (indx+1) * 90
            x_range_min = centre-90
            x_range_max = centre+90

            data = scattering_event[x_range_min : x_range_max]

            data_cropped.append(data)

        return data_cropped

    def _limit_by_constant_angle(self, scattered_data_list, acceptance_angle):

        angle = acceptance_angle/2

        min_acceptance = 0 - angle
        max_acceptance = 0 + angle

        x_range = np.arange(-90, 90, 1)
        min_index_list = np.where(x_range < min_acceptance)
        max_index_list = np.where(x_range > max_acceptance)

        for indx, scatter in enumerate(scattered_data_list):

            scatter[min_index_list] = 0
            scatter[max_index_list] = 0

        return scattered_data_list

    def _area_ratio_change(self, area_sum_list):
        ratio_list = []
        for n in range(len(area_sum_list)):

            if n != 0:
                ratio = area_sum_list[n]/area_sum_list[n-1]

                ratio_list.append(ratio)
        return ratio_list
