# -*- coding: utf-8 -*-

"""
Code to take template spectra, used for RV fitting, and pass them through 4FS to resample them to 4MOST's resolution.
It then further resamples each arm onto a fixed logarithmic stride.
"""

import argparse
import hashlib
import logging
import numpy as np
import os
from os import path as os_path

from fourgp_fourfs import FourFS
from fourgp_degrade.resample import SpectrumResampler
from fourgp_degrade import SpectrumProperties
from fourgp_speclib import SpectrumLibrarySqlite


def command_line_interface(root_path):
    """
    A simple command-line interface for running a tool to resample a library of template spectra onto fixed
    logarithmic rasters representing each of the 4MOST arms.

    We use the python argparse module to build the interface, and return the inputs supplied by the user.

    :param root_path:
        The root path of this 4GP installation; the directory where we can find 4FS.

    :return:
        An object containing the arguments supplied by the user.
    """
    # Read input parameters
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument('--templates-in',
                        required=False,
                        default='turbospec_rv_templates',
                        dest='templates_in',
                        help="Library of spectra to use as templates for RV code")
    parser.add_argument('--workspace', dest='workspace', default="",
                        help="Directory where we expect to find spectrum libraries")
    parser.add_argument('--templates-out',
                        required=False,
                        default="rv_templates_resampled",
                        dest="templates_out",
                        help="Library into which to place resampled templates for RV code")
    parser.add_argument('--binary-path',
                        required=False,
                        default=root_path,
                        dest="binary_path",
                        help="Specify a directory where 4FS binary package is installed")
    args = parser.parse_args()

    # Set up logger
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s:%(filename)s:%(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S')
    logger = logging.getLogger(__name__)

    logger.info("Resampling template spectra")

    return args


def logarithmic_raster(lambda_min, lambda_max, lambda_step):
    """
    Create a logarithmic raster with a fixed logarithmic stride, based on a starting wavelength, finishing wavelength,
    and a mean wavelength step.

    :param lambda_min:
        Smallest wavelength in raster.
    :param lambda_max:
        Largest wavelength in raster.
    :param lambda_step:
        The approximate pixel size in the raster.
    :return:
        A numpy array containing a wavelength raster with fixed logarithmic stride.
    """
    return np.exp(np.arange(
        np.log(lambda_min),
        np.log(lambda_max),
        np.log(1 + lambda_step / lambda_min)
    ))


def resample_templates(args, logger):
    """
    Resample a spectrum library of templates onto a fixed logarithmic stride, representing each of the 4MOST arms in
    turn. We use 4FS to down-sample the templates to the resolution of 4MOST observations, and automatically detect
    the list of arms contained within each 4FS mock observation. We then resample the 4FS output onto a new raster
    with fixed logarithmic stride.

    :param args:
        Object containing arguments supplied by the used, for example the name of the spectrum libraries we use for
        input and output. The required fields are defined by the user interface above.
    :param logger:
        A python logging object.
    :return:
        None.
    """
    # Set path to workspace where we expect to find libraries of spectra
    workspace = args.workspace if args.workspace else os_path.join(args.our_path, "../../../workspace")

    # Open input template spectra
    spectra = SpectrumLibrarySqlite.open_and_search(
        library_spec=args.templates_in,
        workspace=workspace,
        extra_constraints={"continuum_normalised": 0}
    )

    templates_library, templates_library_items, templates_spectra_constraints = \
        [spectra[i] for i in ("library", "items", "constraints")]

    # Create new SpectrumLibrary to hold the resampled output templates
    library_path = os_path.join(workspace, args.templates_out)
    output_library = SpectrumLibrarySqlite(path=library_path, create=True)

    # Instantiate 4FS wrapper
    etc_wrapper = FourFS(
        path_to_4fs=os_path.join(args.binary_path, "OpSys/ETC"),
        snr_list=[250.],
        magnitude=13,
        snr_per_pixel=True
    )

    for input_spectrum_id in templates_library_items:
        logger.info("Working on <{}>".format(input_spectrum_id['filename']))

        # Open Spectrum data from disk
        input_spectrum_array = templates_library.open(ids=input_spectrum_id['specId'])

        # Load template spectrum (flux normalised)
        template_flux_normalised = input_spectrum_array.extract_item(0)

        # Look up the unique ID of the star we've just loaded
        # Newer spectrum libraries have a uid field which is guaranteed unique; for older spectrum libraries use
        # Starname instead.

        # Work out which field we're using (uid or Starname)
        spectrum_matching_field = 'uid' if 'uid' in template_flux_normalised.metadata else 'Starname'

        # Look up the unique ID of this object
        object_name = template_flux_normalised.metadata[spectrum_matching_field]

        # Search for the continuum-normalised version of this same object (which will share the same uid / name)
        search_criteria = {
            spectrum_matching_field: object_name,
            'continuum_normalised': 1
        }

        continuum_normalised_spectrum_id = templates_library.search(**search_criteria)

        # Check that continuum-normalised spectrum exists and is unique
        assert len(continuum_normalised_spectrum_id) == 1, "Could not find continuum-normalised spectrum."

        # Load the continuum-normalised version
        template_continuum_normalised_arr = templates_library.open(
            ids=continuum_normalised_spectrum_id[0]['specId']
        )

        # Turn the SpectrumArray we got back into a single Spectrum
        template_continuum_normalised = template_continuum_normalised_arr.extract_item(0)

        # Now create a mock observation of this template using 4FS
        logger.info("Passing template through 4FS")
        mock_observed_template = etc_wrapper.process_spectra(
            spectra_list=((template_flux_normalised, template_continuum_normalised),)
        )

        # Loop over LRS and HRS
        for mode in mock_observed_template:

            # Loop over the spectra we simulated (there was only one!)
            for index in mock_observed_template[mode]:

                # Loop over the various SNRs we simulated (there was only one!)
                for snr in mock_observed_template[mode][index]:

                    # Create a unique ID for this arm's data
                    unique_id = hashlib.md5(os.urandom(32)).hexdigest()[:16]

                    # Import the flux- and continuum-normalised spectra separately, but give them the same ID
                    for spectrum_type in mock_observed_template[mode][index][snr]:

                        # Extract continuum-normalised mock observation
                        logger.info("Resampling {} spectrum".format(mode))
                        mock_observed = mock_observed_template[mode][index][snr][spectrum_type]

                        # Replace errors which are nans with a large value
                        mock_observed.value_errors[np.isnan(mock_observed.value_errors)] = 1000.

                        # Check for NaN values in spectrum itself
                        if not np.all(np.isfinite(mock_observed.values)):
                            print("Warning: NaN values in template <{}>".format(template_flux_normalised.metadata['Starname']))
                            mock_observed.value_errors[np.isnan(mock_observed.values)] = 1000.
                            mock_observed.values[np.isnan(mock_observed.values)] = 1.

                        # Resample template onto a logarithmic raster of fixed step
                        resampler = SpectrumResampler(mock_observed)

                        # Construct the raster for each wavelength arm
                        wavelength_arms = SpectrumProperties(mock_observed.wavelengths).wavelength_arms()

                        # Resample 4FS output for each arm onto a fixed logarithmic stride
                        for arm_count, arm in enumerate(wavelength_arms["wavelength_arms"]):
                            arm_raster, mean_pixel_width = arm
                            name = "{}_{}".format(mode, arm_count)

                            arm_info = {
                                "lambda_min": arm_raster[0],
                                "lambda_max": arm_raster[-1],
                                "lambda_step": mean_pixel_width
                            }

                            arm_raster = logarithmic_raster(lambda_min=arm_info['lambda_min'],
                                                            lambda_max=arm_info['lambda_max'],
                                                            lambda_step=arm_info['lambda_step']
                                                            )

                            # Resample 4FS output onto a fixed logarithmic step
                            mock_observed_arm = resampler.onto_raster(arm_raster)

                            # Save it into output spectrum library
                            output_library.insert(spectra=mock_observed_arm,
                                                  filenames=input_spectrum_id['filename'],
                                                  metadata_list={
                                                      "uid": unique_id,
                                                      "template_id": object_name,
                                                      "mode": mode,
                                                      "arm_name": "{}_{}".format(mode,arm_count),
                                                      "lambda_min": arm_raster[0],
                                                      "lambda_max": arm_raster[-1],
                                                      "lambda_step": mean_pixel_width
                                                  })
