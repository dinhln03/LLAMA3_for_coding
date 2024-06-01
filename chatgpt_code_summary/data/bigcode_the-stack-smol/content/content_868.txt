import os
import numpy as np
import rasterio


aggregate_forest = np.vectorize(lambda x: np.where(0 < x < 6, 1, x))
aggregate_agriculture = np.vectorize(lambda x: np.where(11 < x < 21, 21, x))


for dirs, subdirs, files in os.walk('../output/ceara/'):
    for file in files:
        wp_raster = rasterio.open('../output/ceara/' + file)

        file_name = file.replace('id_', '')
        wp_id = int(file_name.replace('.tif', ''))

        out_raster_temp = aggregate_forest(wp_raster.read(range(1, 34)))
        out_raster = aggregate_agriculture(out_raster_temp)
        out_raster = out_raster.astype('uint8')
        out_meta = wp_raster.meta

        with rasterio.open('../output/ceara_agg_v2/' + 'agg_v2_id_' + str(wp_id) + '.tif', 'w', **out_meta) as raster:
            raster.write(out_raster)
