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

