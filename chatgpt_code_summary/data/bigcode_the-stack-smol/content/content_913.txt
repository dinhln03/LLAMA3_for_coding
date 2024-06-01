import os
import numpy as np
from netCDF4 import Dataset


def load_region(region_id, local=False, return_regions=False):

    if local:
        _vr = Dataset(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), r"data/terrain_parameters/VarslingsOmr_2017.nc"),
            "r")
        # flip up-down because Meps data is upside down
        #_regions = np.flipud(_vr.variables["LokalOmr_2018"][:])
        _regions = _vr.variables["LokalOmr_2018"][:]
    else:
        _vr = Dataset(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), r"data/terrain_parameters/VarslingsOmr_2019.nc"),
            "r")
        # flip up-down because Meps data is upside down
        #_regions = np.flipud(_vr.variables["skredomr19_km"][:])
        _regions = _vr.variables["skredomr19_km"][:]
        print("Missing value: {mv}".format(mv=_vr.variables["skredomr19_km"].missing_value))

    _region_bounds = np.where(_regions == region_id)  # just to get the bounding box

    # get the lower left and upper right corner of a rectangle around the region
    y_min, y_max, x_min, x_max = min(_region_bounds[0].flatten()), max(_region_bounds[0].flatten()), \
                                 min(_region_bounds[1].flatten()), max(_region_bounds[1].flatten())

    #reg_mask = np.ma.masked_where(_regions[y_min:y_max, x_min:x_max] == region_id, _regions[y_min:y_max, x_min:x_max]).mask
    #reg_mask = np.where(_regions[y_min:y_max, x_min:x_max] == region_id, _regions[y_min:y_max, x_min:x_max], np.nan)
    reg_mask = np.where(_regions[y_min:y_max, x_min:x_max] == region_id, 1., np.nan)
    #reg_mask = np.ma.masked_where(_reg_mask == region_id).mask
    _vr.close()

    if return_regions:
        return _regions, reg_mask, y_min, y_max, x_min, x_max
    else:
        return reg_mask, y_min, y_max, x_min, x_max


def clip_region(nc_variable, region_mask, t_index, y_min, y_max, x_min, x_max):
    s = len(nc_variable.shape)

    if s == 2:
        #return np.flipud(region_mask * nc_variable[y_min:y_max, x_min:x_max])
        return (region_mask * nc_variable[y_min:y_max, x_min:x_max])
    elif s == 3:
        #return np.flipud(region_mask * nc_variable[t_index, y_min:y_max, x_min:x_max])
        return (region_mask * nc_variable[t_index, y_min:y_max, x_min:x_max])
    elif s == 4:
        #return np.flipud(region_mask * nc_variable[t_index, 0, y_min:y_max, x_min:x_max])
        return (region_mask * nc_variable[t_index, 0, y_min:y_max, x_min:x_max])
    else:
        print('Input array needs to have 2- to 4-dimensions: {0} were given.'.format(s))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    regions, region_mask, y_min, y_max, x_min, x_max = load_region(3013, return_regions=True)
    print(region_mask, type(region_mask), np.unique(region_mask))
    clp = clip_region(regions, region_mask, 0, y_min, y_max, x_min, x_max)
    plt.imshow(clp)
    plt.show()

    k = 'm'