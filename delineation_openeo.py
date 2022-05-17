'''
Created on Jun 9, 2020

@author: banyait
'''
import openeo
import logging
import os
import pyproj
import numpy
import json
from pathlib import Path
import geopandas as gpd
import utm
import shapely
from delineation_functions import window_around_centroid, create_input_NDVI_cube\
    , _do_image_selection, _do_segmentation, _do_vectorization,_get_epsg, _do_vectorization_watershed


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


def load_udf(relative_path):
    with open(str(Path(relative_path)), 'r+') as f:
        return f.read()


def zone(coordinates):
    if 56 <= coordinates[1] < 64 and 3 <= coordinates[0] < 12:
        return 32
    if 72 <= coordinates[1] < 84 and 0 <= coordinates[0] < 42:
        if coordinates[0] < 9:
            return 31
        elif coordinates[0] < 21:
            return 33
        elif coordinates[0] < 33:
            return 35
        return 37
    return int((coordinates[0] + 180) / 6) + 1


if __name__ == '__main__':
    openeo_url = 'openeo-dev.vito.be'
    modified_prediction = False

    basedir = r'/data/users/Private/bontek/git/e-shape/Cases/Ukraine/field delineation/Stratified_test_100_points/Results/openeo/test/TERRA'

    year = 2021
    windowsize = 256

    startdate = str(year - 1) + '-10-01'
    enddate = str(year) + '-08-31'

    job_options = {
        'driver-memory': '8G',
        'executor-memory': '8G'
    }

    eoconn = openeo.connect(openeo_url)
    eoconn.authenticate_basic('bontek', 'bontek123')


    #### Load the patches that are needed to do the prediction:
    shp = gpd.read_file(r'/data/users/Private/bontek/git/e-shape/Cases/Ukraine/field delineation/Stratified_test_100_points/Shapes/Ukraine_fielddelineation_random_test_locations.shp')
    Point_geom = shp.iloc[1].geometry
    Point_geom = shapely.geometry.Point(5.00468,51.19924)
    ### Now draw a window of the windowsize around the center point
    polyg_window = window_around_centroid(Point_geom, window_size = windowsize)
    bbox = polyg_window.bounds

    ndvi_cube = create_input_NDVI_cube(startdate, enddate, bbox, eoconn)

    # select top 12 usable layers
    ndvi_cube_filtered = _do_image_selection(ndvi_cube,
                                             modified_filter=modified_prediction)

    # produce the segmentation image
    segmentationband = _do_segmentation(ndvi_cube_filtered, startdate, enddate,
                                        modified_prediction=modified_prediction, size = windowsize)

    vectorization = _do_vectorization(segmentationband, size = windowsize)

    #vectorization = _do_vectorization_watershed(segmentationband, size = windowsize)

    segm_result_json = vectorization.send_job().start_and_wait().get_result().load_json()

    with open(os.path.join(basedir, 'segmentation_result.json'), 'w') as f:
        json.dump(segm_result_json, f)

    with open(os.path.join(basedir, 'segmentation_result.json')) as f:
        polygons = json.load(f)
    from shapely.geometry import shape
    import geopandas as gpd

    utm_zone_nr = utm.from_latlon(Point_geom.y, Point_geom.x)[2]
    epsg_UTM_field = _get_epsg(Point_geom.y, utm_zone_nr)

    geom = [shape(p) for p in polygons]
    gpd.GeoDataFrame(geometry=geom, crs=f"EPSG:{epsg_UTM_field}").to_file(os.path.join(basedir, 'segmentation_polygons.shp'))

    logger.info('FINISHED')






