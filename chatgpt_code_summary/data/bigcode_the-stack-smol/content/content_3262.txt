#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Test the quoted APOGEE uncertainties from individual (rebinned) spectra. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from glob import glob
from itertools import combinations


def get_differences(apStar_filename):

    image = fits.open(apStar_filename)

    N_visits = image[0].header["NVISITS"]

    data_index = 1
    error_index = 2
    mask_index = 3

    # Generate all permutations.
    differences = []
    for i, j in combinations(range(N_visits), 2):

        di = image[data_index].data[i + 2, :]
        dj = image[data_index].data[j + 2, :]
        sigma = np.sqrt(image[error_index].data[i + 2, :]**2 \
            + image[error_index].data[j + 2, :]**2)

        ok = (di > 0) * (dj > 0) * np.isfinite(di * dj * sigma) \
            * (image[mask_index].data[i + 2, :] == 0) \
            * (image[mask_index].data[j + 2, :] == 0)
        differences.extend(((di - dj)/sigma)[ok])

    differences = np.array(differences).flatten()
    return differences


def plot_differences(differences):

    fig, ax = plt.subplots(1)
    y_bin, x_bin, _ = ax.hist(differences, bins=100, facecolor="#666666")
    x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 1000)
    y = np.exp(-0.5*x**2)/np.sqrt(2*np.pi)
    ax.plot(x, y*np.trapz(y_bin, x=x_bin[1:])/np.sqrt(2*np.pi), lw=2, c="r")
    ax.set_title("mu = {0:.1f}, sigma(|d|) = {1:.1f}".format(
        np.median(differences), np.std(np.abs(differences))))

    ax.set_xlabel("(F1 - F2)/sqrt(sigma_1^2 + sigma_2^2)")
    return fig



if __name__ == "__main__":

    filenames = glob("APOGEE/*.fits")
    all_differences = []
    for filename in filenames:

        differences = get_differences(filename)
        if len(differences) > 0:        
            fig = plot_differences(differences)
            fig.savefig("APOGEE/{0}.png".format(filename.split("/")[-1].split(".")[0]))

        plt.close("all")
        print(filename)
        all_differences.extend(differences)

    fig = plot_differences(np.array(all_differences))
    fig.savefig("APOGEE/all.png")


