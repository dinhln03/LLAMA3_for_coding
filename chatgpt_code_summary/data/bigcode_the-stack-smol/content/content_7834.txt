"""
Extract CLOS / NLOS lookup.

Written by Ed Oughton.

March 2021

"""
import os
import configparser
import json
import math
import glob
import random
import numpy as np
import pandas as pd
import geopandas as gpd
import pyproj
from shapely.geometry import Point, Polygon, box, LineString
from shapely.ops import transform
import rasterio
# import networkx as nx
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from rasterstats import zonal_stats, gen_zonal_stats
from tqdm import tqdm

grass7bin = r'"C:\Program Files\GRASS GIS 7.8\grass78.bat"'
os.environ['GRASSBIN'] = grass7bin
os.environ['PATH'] += ';' + r"C:\Program Files\GRASS GIS 7.8\lib"

from grass_session import Session
from grass.script import core as gcore

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), "script_config.ini"))
BASE_PATH = CONFIG["file_locations"]["base_path"]

DATA_RAW = os.path.join(BASE_PATH, "raw")
DATA_INTERMEDIATE = os.path.join(BASE_PATH, "intermediate")
DATA_PROCESSED = os.path.join(BASE_PATH, "processed")


def load_raster_tile_lookup(iso3):
    """
    Load in the preprocessed raster tile lookup.

    Parameters
    ----------
    iso3 : string
        Country iso3 code.

    Returns
    -------
    lookup : dict
        A lookup table containing raster tile boundary coordinates
        as the keys, and the file paths as the values.

    """
    path = os.path.join(DATA_INTERMEDIATE, iso3, 'raster_lookup.csv')
    data = pd.read_csv(path)
    data = data.to_records('dicts')

    lookup = {}

    for item in data:

        coords = (item['x1'], item['y1'], item['x2'], item['y2'])

        lookup[coords] = item['path']

    return lookup


def generate_grid(iso3, side_length):
    """
    Generate a spatial grid for the chosen country.
    """
    directory = os.path.join(DATA_INTERMEDIATE, iso3, 'grid')

    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = 'grid_{}_{}_km.shp'.format(side_length, side_length)
    path_output = os.path.join(directory, filename)

    if os.path.exists(path_output):
        return

    filename = 'national_outline.shp'
    path = os.path.join(DATA_INTERMEDIATE, iso3, filename)
    country_outline = gpd.read_file(path, crs="epsg:4326")

    country_outline.crs = "epsg:4326"
    country_outline = country_outline.to_crs("epsg:3857")

    xmin, ymin, xmax, ymax = country_outline.total_bounds

    polygons = manually_create_grid(
        xmin, ymin, xmax, ymax, side_length, side_length
    )

    grid = gpd.GeoDataFrame({'geometry': polygons}, crs="epsg:3857")#[:100]

    intersection = gpd.overlay(grid, country_outline, how='intersection')
    intersection.crs = "epsg:3857"
    intersection['area_km2'] = intersection['geometry'].area / 1e6
    intersection = intersection.to_crs("epsg:4326")
    intersection.to_file(path_output, crs="epsg:4326")

    return intersection


def manually_create_grid(xmin, ymin, xmax, ymax, length, wide):
    """

    """
    cols = list(range(int(np.floor(xmin)), int(np.ceil(xmax - int(wide))), int(wide)))
    rows = list(range(int(np.floor(ymin)), int(np.ceil(ymax)), int(length)))

    polygons = []

    for x in cols:
        for y in rows:
            polygons.append(
                Polygon([(x, y), (x+wide, y), (x+wide, y-length), (x, y-length)])
            )

    return polygons


def find_tile(polygon, tile_lookup):
    """

    Parameters
    ----------
    polygon : tuple
        The bounds of the modeling region.
    tile_lookup : dict
        A lookup table containing raster tile boundary coordinates
        as the keys, and the file paths as the values.

    Return
    ------
    output : list
        Contains the file path to the correct raster tile. Note:
        only the first element is returned and if there are more than
        one paths, an error is returned.

    """
    output = []

    poly_bbox = box(polygon[0], polygon[1], polygon[2], polygon[3])

    for key, value in tile_lookup.items():

        bbox = box(key[0], key[1], key[2], key[3])

        if bbox.intersects(poly_bbox):
            output.append(value)

    if len(output) == 1:
        return output[0]
    elif len(output) > 1:
        print('Problem with find_tile returning more than 1 path')
        return output[0]
    else:
        print('Problem with find_tile: Unable to find raster path')


def add_id_range_data_to_grid(iso3, tile_lookup, side_length):
    """
    Query the Digital Elevation Model to get an estimated interdecile
    range for each grid square.

    """
    directory = os.path.join(DATA_INTERMEDIATE, iso3, 'grid')
    filename = 'grid_final.shp'
    path_output = os.path.join(directory, filename)

    if os.path.exists(path_output):
        return gpd.read_file(path_output, crs='epsg:4328')

    filename = 'grid_{}_{}_km.shp'.format(side_length, side_length)
    path = os.path.join(directory, filename)
    grid = gpd.read_file(path, crs='epsg:4328')

    output = []

    for idx, grid_tile in grid.iterrows():

        path_input = find_tile(
            grid_tile['geometry'].bounds,
            tile_lookup
        )

        stats = next(gen_zonal_stats(
            grid_tile['geometry'],
            path_input,
            add_stats={
                'interdecile_range': interdecile_range
            },
            nodata=0
        ))

        id_range_m = stats['interdecile_range']

        output.append({
            'type': 'Feature',
            'geometry': grid_tile['geometry'],
            'properties': {
                'id_range_m': id_range_m,
                'area_km2': grid_tile['area_km2'],
                # 'pop_density_km2': grid_tile['pop_densit'],
                # 'population': grid_tile['population'],
            }
        })

    output = gpd.GeoDataFrame.from_features(output, crs='epsg:4326')

    output = output.replace([np.inf, -np.inf], np.nan)

    output = output[output.geometry.notnull()]

    output.to_file(path_output, crs="epsg:4326")

    return output


def interdecile_range(x):
    """
    Get range between bottom 10% and top 10% of values.

    This is from the Longley-Rice Irregular Terrain Model.

    Code here: https://github.com/edwardoughton/itmlogic
    Paper here: https://joss.theoj.org/papers/10.21105/joss.02266.pdf

    Parameters
    ----------
    x : list
        Terrain profile values.

    Returns
    -------
    interdecile_range : int
        The terrain irregularity parameter.

    """
    q90, q10 = np.percentile(x, [90, 10])

    interdecile_range = int(round(q90 - q10, 0))

    return interdecile_range


def estimate_terrain_deciles(grid):
    """

    """
    # terrain_lookup = grid.loc[grid['area_km2'] > 1000].reset_index()

    terrain_lookup = grid
    terrain_lookup['decile'] = pd.qcut(terrain_lookup['id_range_m'], 10, labels=False)

    terrain_lookup = terrain_lookup[['decile', 'id_range_m']]

    terrain_lookup = terrain_lookup.groupby(['decile']).min()

    terrain_lookup = terrain_lookup['id_range_m'].to_list()

    return terrain_lookup


def select_grid_sampling_areas(iso3, grid, lut):
    """

    """
    for i in range(1, 11):
        if i == 1:
            grid.loc[(grid['id_range_m'] < lut[1]), 'decile'] = str(i)
            value_name = '0-{}'.format(str(lut[1]))
            grid.loc[(grid['id_range_m'] < lut[1]), 'value'] = value_name
        elif i <= 9:
            grid.loc[(
                grid['id_range_m'] >= lut[i-1]) &
                (grid['id_range_m'] <= lut[i]), 'decile'] = str(i)
            value_name = '{}-{}'.format(str(lut[i-1]), str(lut[i]))
            grid.loc[(
                grid['id_range_m'] >= lut[i-1]) &
                (grid['id_range_m'] <= lut[i]), 'value'] = value_name
        elif i == 10:
            grid.loc[(grid['id_range_m'] > lut[i-1]), 'decile'] = str(i)
            value_name = '>{}'.format(str(lut[i-1]))
            grid.loc[(grid['id_range_m'] > lut[i-1]), 'value'] = value_name
        else:
            continue

    np.random.seed(2)

    grid = grid.loc[grid['area_km2'] > 2400].reset_index()

    sampling_areas = grid.groupby(['decile']).apply(lambda x: x.sample(1)).reset_index(drop=True)

    directory = os.path.join(DATA_INTERMEDIATE, iso3, 'sampling_area')

    if not os.path.exists(directory):
        os.makedirs(directory)

    sampling_areas.to_file(os.path.join(directory, 'sampling_areas.shp'))

    sampling_areas.crs = 'epsg:4326'

    return sampling_areas


def get_points(iso3, sampling_areas, tile_lookup, point_spacing):
    """

    """
    directory = os.path.join(DATA_INTERMEDIATE, iso3, 'sampling_points')

    if not os.path.exists(directory):
        os.makedirs(directory)

    sampling_areas = sampling_areas.to_crs("epsg:3857")

    for idx, sampling_area in sampling_areas.iterrows():

        lon = sampling_area['geometry'].representative_point().coords[0][0]
        lat = sampling_area['geometry'].representative_point().coords[0][1]
        filename = "{}-{}".format(lon, lat)

        xmin, ymin, xmax, ymax = sampling_area['geometry'].bounds

        polygons = manually_create_grid(xmin, ymin, xmax, ymax, point_spacing, point_spacing)

        #make geopandas dataframes
        grid_sample = gpd.GeoDataFrame({'geometry': polygons}, crs="epsg:3857")
        boundary = gpd.GeoDataFrame({'geometry': sampling_area['geometry']},
                    crs="epsg:3857", index=[0])

        #only get points within the tile boundary
        grid_sample = gpd.overlay(grid_sample, boundary, how='intersection')

        grid_sample = grid_sample.to_crs("epsg:4326") #convert to lon lat

        ##get the highest points in each grid sample tile
        sampling_points = find_points(iso3, grid_sample, tile_lookup, filename)#[:1]

        ##convert to projected for viewsheding
        sampling_points = sampling_points.to_crs("epsg:4326")

        path_output = os.path.join(directory, filename + '.shp')
        sampling_points.to_file(path_output)

    return sampling_points


def find_points(iso3, grid_sample, tile_lookup, filename):
    """

    """
    filename_2 = filename + '.shp'
    directory = os.path.join(DATA_INTERMEDIATE, iso3, 'sampling_points')
    path_output = os.path.join(directory, filename_2)

    if os.path.exists(path_output):
        return gpd.read_file(path_output, crs='epsg:4326')

    output = []

    for idx, grid_tile in grid_sample.iterrows():

        min_x, min_y, max_x, max_y = grid_tile['geometry'].bounds

        geom = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))

        output.append({
            'type': 'Feature',
            'geometry': geom,
            'properties': {
            }
        })

    output = gpd.GeoDataFrame.from_features(output, crs='epsg:4326')

    return output


def generate_viewsheds(iso3, sampling_areas, sampling_points):
    """

    """
    sampling_areas = sampling_areas.to_crs("epsg:3857")

    #set output folder
    folder_out_viewsheds = os.path.join(DATA_INTERMEDIATE, iso3, 'viewsheds')

    if not os.path.exists(folder_out_viewsheds):
        os.makedirs(folder_out_viewsheds)

    for idx, sampling_area in tqdm(sampling_areas.iterrows(),
        total=sampling_areas.shape[0]):

        output = []

        lon = sampling_area['geometry'].representative_point().coords[0][0]
        lat = sampling_area['geometry'].representative_point().coords[0][1]
        area_filename = "{}-{}".format(lon, lat)
        print('--Working on {}'.format(area_filename))

        ##load sampling points
        directory = os.path.join(DATA_INTERMEDIATE, iso3, 'sampling_points')
        points = gpd.read_file(os.path.join(directory, area_filename + '.shp'))#[:2]

        ##convert to lon lat to get correct raster tile
        sampling_area_df = gpd.GeoDataFrame({'geometry': sampling_area['geometry']},
                            crs="epsg:3857", index=[0])
        sampling_area_df = sampling_area_df.to_crs("epsg:4326")

        for idx, item in sampling_area_df.iterrows():
            #needs a loop because the data structure needs a series
            path_input = find_tile(item['geometry'].bounds, tile_lookup)

        for idx, point in tqdm(points.iterrows(), total=points.shape[0]):

            results = []

            lon = point['geometry'].representative_point().coords[0][0]
            lat = point['geometry'].representative_point().coords[0][1]
            filename2 = "{}-{}".format(lon, lat)

            path_output = os.path.join(folder_out_viewsheds, filename2)

            file_path = os.path.join(path_output, 'location', 'PERMANENT',
                'viewsheds', filename2 + '.tif')

            x = point['geometry'].coords[0][0]
            y = point['geometry'].coords[0][1]

            if not os.path.exists(file_path):
                try:
                    viewshed((x, y), path_input, path_output, filename2, 45000, 'epsg:4326')
                except:
                    print('--Viewshed already exists')

            seen = set()

            for idx, node in tqdm(points.iterrows(), total=points.shape[0]):

                x2 = node['geometry'].coords[0][0]
                y2 = node['geometry'].coords[0][1]

                link = '{}_{}_{}_{}'.format(x, y, x2, y2)

                if link in seen:
                    continue

                dist = find_distance((x, y), (x2, y2))

                if dist < 10:
                    continue

                los = check_los(file_path, (x2, y2))

                results.append({
                    'sampling_area': area_filename,
                    'point_id': filename2,
                    'node_id': '{}_{}'.format(x2, y2),
                    'distance': dist,
                    'id_range_m': sampling_area['id_range_m'],
                    'decile': sampling_area['decile'],
                    'los': los,
                })

                seen.add('{}_{}_{}_{}'.format(x, y, x2, y2))
                seen.add('{}_{}_{}_{}'.format(x2, y2, x, y))

            output = output + results

        output = pd.DataFrame(output)
        folder = os.path.join(DATA_INTERMEDIATE, iso3, 'los_results')

        if not os.path.exists(folder):
            os.makedirs(folder)

        output.to_csv(os.path.join(folder, area_filename + '.csv'), index=False)


def viewshed(point, path_input, path_output, tile_name, max_distance, crs):
    """
    Perform a viewshed using GRASS.

    Parameters
    ---------
    point : tuple
        The point being queried.
    tile_lookup : dict
        A lookup table containing raster tile boundary coordinates
        as the keys, and the file paths as the values.
    path_output : string
        The directory path for the output folder.
    tile_name : string
        The name allocated to the viewshed tile.
    max_distance : int
        The maximum distance a path can be.
    crs : string
        The coordinate reference system in use.

    Returns
    -------
    grid : dataframe
        A geopandas dataframe containing the created grid.

    """
    with Session(gisdb=path_output, location="location", create_opts=crs):

        # print('parse command')
        # print(gcore.parse_command("g.gisenv", flags="s"))#, set="DEBUG=3"

        # print('r.external')
        # now link a GDAL supported raster file to a binary raster map layer,
        # from any GDAL supported raster map format, with an optional title.
        # The file is not imported but just registered as GRASS raster map.
        gcore.run_command('r.external', input=path_input, output=tile_name, overwrite=True)

        # print('r.external.out')
        #write out as geotiff
        gcore.run_command('r.external.out', directory='viewsheds', format="GTiff")

        # print('r.region')
        #manage the settings of the current geographic region
        gcore.run_command('g.region', raster=tile_name)

        # print('r.viewshed')
        #for each point in the output that is NULL: No LOS
        gcore.run_command('r.viewshed', #flags='e',
                input=tile_name,
                output='{}.tif'.format(tile_name),
                coordinate= [point[0], point[1]],
                observer_elevation=30,
                target_elevation=30,
                memory=5000,
                overwrite=True,
                quiet=True,
                max_distance=max_distance,
                # verbose=True
        )


def check_los(path_input, point):
    """
    Find potential LOS high points.

    Parameters
    ----------
    path_input : string
        File path for the digital elevation raster tile.
    point : tuple
        Coordinate point being queried.

    Returns
    -------
    los : string
        The Line of Sight (los) of the path queried.

    """
    with rasterio.open(path_input) as src:

        x = point[0]
        y = point[1]

        for val in src.sample([(x, y)]):

            if np.isnan(val):
                # print('is nan: {} therefore nlos'.format(val))
                los = 'nlos'
                return los
            else:
                # print('is not nan: {} therefore los'.format(val))
                los ='clos'
                return los


def find_distance(point1, point2):
    """

    """
    point1 = Point(point1)
    point1 = gpd.GeoDataFrame({'geometry': [point1]}, index=[0])
    point1 = point1.set_crs('epsg:4326')
    point1 = point1.to_crs('epsg:3857')

    point2 = Point(point2)
    point2 = gpd.GeoDataFrame({'geometry': [point2]}, index=[0])
    point2 = point2.set_crs('epsg:4326')
    point2 = point2.to_crs('epsg:3857')

    dist = LineString([
        (point1['geometry'][0].coords[0][0], point1['geometry'][0].coords[0][1]),
        (point2['geometry'][0].coords[0][0], point2['geometry'][0].coords[0][1])
    ]).length

    return dist


def collect_results(iso3, sampling_areas):
    """

    """
    sampling_areas = sampling_areas.to_crs("epsg:3857")#[:1]

    output = []

    #set output folder
    for idx, sampling_area in sampling_areas.iterrows():

        lon = sampling_area['geometry'].representative_point().coords[0][0]
        lat = sampling_area['geometry'].representative_point().coords[0][1]
        filename = "{}-{}".format(lon, lat)

        directory = os.path.join(DATA_INTERMEDIATE, iso3, 'los_results')
        data = pd.read_csv(os.path.join(directory, filename + '.csv'))

        seen = set()
        interval_size = 2500

        for distance_lower in range(0, 45000, interval_size):

            distance_upper = distance_lower + interval_size

            clos = 0
            nlos = 0

            for idx, item in data.iterrows():

                path_id = '{}_{}_{}'.format(
                    item['point_id'],
                    item['node_id'],
                    item['distance']
                )

                if not path_id in seen:
                    if item['distance'] < distance_upper:

                        if item['los'] == 'clos':
                            clos += 1
                        elif item['los'] == 'nlos':
                            nlos += 1
                        else:
                            print('Did not recognize los')

                        seen.add(path_id)


            if clos > 0:
                clos_probability = (clos / (clos + nlos))
            else:
                clos_probability = 'no data'

            if nlos > 0:
                nlos_probability = (nlos / (clos + nlos))
            else:
                nlos_probability = 'no data'

            output.append({
                'decile': item['decile'],
                'id_range_m': item['id_range_m'],
                'distance_lower': distance_lower,
                'distance_upper': distance_upper,
                'total_samples': clos + nlos,
                'clos_probability': clos_probability,
                'nlos_probability': nlos_probability,
            })

    output = pd.DataFrame(output)
    folder = os.path.join(DATA_INTERMEDIATE, iso3)
    output.to_csv(os.path.join(folder, 'los_lookup.csv'), index=False)


if __name__ == "__main__":

    countries = [
        ("PER", 5e4, 25e2),
        ("IDN", 5e4, 25e2),
    ]

    for country in countries:

        iso3 = country[0]
        side_length = country[1]
        point_spacing = country[2]

        ##Load the raster tile lookup
        tile_lookup = load_raster_tile_lookup(iso3)

        ##Generate grids
        generate_grid(iso3, side_length) #1e5

        # ##Add interdecile range to grid
        grid = add_id_range_data_to_grid(iso3, tile_lookup, side_length)

        ##Get the terrain deciles
        terrain_values = estimate_terrain_deciles(grid)

        ##Get the grid tile samples
        sampling_areas = select_grid_sampling_areas(iso3, grid, terrain_values)#[:1]

        ##Generate the terrain lookup
        sampling_points = get_points(iso3, sampling_areas, tile_lookup, point_spacing)#[:1]

        ##Process viewsheds
        generate_viewsheds(iso3, sampling_areas, sampling_points)

        ## Collect results
        collect_results(iso3, sampling_areas)
