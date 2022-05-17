import os
import pyproj
import shapely
from pathlib import Path
from openeo.internal.graph_building import PGNode

def _get_epsg(lat, zone_nr):
    if lat >= 0:
        epsg_code = '326' + str(zone_nr)
    else:
        epsg_code = '327' + str(zone_nr)
    return int(epsg_code)


def load_udf(relative_path):
    with open(str(Path(relative_path)), 'r+') as f:
        return f.read()


def create_mask_cropland(eoconn, LC_collection, bbox):
    #Function to mask OpenEO collection for WorldCover agriculture

    LC_info = eoconn.load_collection(LC_collection, bands=['MAP'],
                                     spatial_extent=dict(zip(["west", "south", "east", "north"], bbox)),
                                         temporal_extent=["2000-06-07", "2021-06-07"])
    mask = LC_info.band("MAP") != 40

    return mask

def create_input_NDVI_cube(startdate, enddate, bbox, eoconn, mask_LC = False):
    s2_bands = eoconn.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=[startdate, enddate],
        spatial_extent=dict(zip(["west", "south", "east", "north"], bbox)),
        bands=["B04", "B08", "SCL"]
    )
    s2_bands = s2_bands.process("mask_scl_dilation", data=s2_bands, scl_band_name="SCL")

    # # Composite 10-daily
    # s2_bands = s2_bands.aggregate_temporal_period(period="dekad",
    #                                             reducer="median")
    #
    # # Linearly interpolate missing values
    # s2_bands = s2_bands.apply_dimension(dimension="t",
    #                                   process="array_interpolate_linear")

    ndviband = s2_bands.ndvi(red="B04", nir="B08")

    if mask_LC:
        # Create the agricultural mask
        mask_WC = create_mask_cropland(eoconn, 'ESA_WORLDCOVER_10M_2020_V1', bbox)
        # Apply the mask to BSI
        ndviband = ndviband.mask(mask_WC.resample_cube_spatial(ndviband))

    return ndviband

def _do_image_selection(input_cube, name_function = 'udf_reduce_images', basedir = os.getcwd(), modified_filter= False):
    input_cube_filtered = input_cube.apply_dimension(
        lambda data: data.run_udf(udf=load_udf(Path(basedir).joinpath(f'{name_function}.py')), runtime='Python',
                                  context={'modified_filter': modified_filter}),
        dimension = 't')

    return input_cube_filtered

def _do_segmentation(input_cube, startdate, enddate,size = 64, overlap = 32, basedir = os.getcwd(), name_function = 'segmentation',  modified_prediction = False, patch_size = 128):
    segm_cube = input_cube.apply_neighborhood(
        lambda data: data.run_udf(udf=load_udf(Path(basedir).joinpath(f'{name_function}.py')), runtime='Python',
                                  context={
                                      'startdate': startdate,
                                      'enddate': enddate,
                                      'modified_prediction': modified_prediction,
                                      'patch_size': patch_size
                                  }),
        size=[
            {'dimension': 'x', 'value': size, 'unit': 'px'},
            {'dimension': 'y', 'value': size, 'unit': 'px'}
        ],
        overlap=[
            {'dimension': 'x', 'value': overlap, 'unit': 'px'},
            {'dimension': 'y', 'value': overlap, 'unit': 'px'}
        ]
    )
    return segm_cube


def _do_vectorization(segmentation_cube, size = 512, basedir = os.getcwd(), overlap = 0, name_function = 'udf_sobel_felzenszwalb', conversion_vector = True):
    segmentation_cube = segmentation_cube.apply_neighborhood(
        lambda data: data.run_udf(udf=load_udf(Path(basedir).joinpath(f'{name_function}.py')), runtime='Python'),
        size=[
            {'dimension': 'x', 'value': 128, 'unit': 'px'},
            {'dimension': 'y', 'value': 128, 'unit': 'px'}
        ],
        overlap=[
            {'dimension': 'x', 'value': 64, 'unit': 'px'},
            {'dimension': 'y', 'value': 64, 'unit': 'px'}
        ]
    )

    # segmentation_cube = segmentation_cube.apply_dimension(
    #     lambda data: data.run_udf(udf=load_udf(Path(basedir).joinpath(f'{name_function}.py')), runtime='Python',
    #                               context={}),
    #     dimension = 't')
    # vectorization
    if conversion_vector:
        vectorization = segmentation_cube.raster_to_vector()
    else:
        vectorization = segmentation_cube

    return vectorization

def _do_vectorization_watershed(segmentation_cube, size = 512, basedir = os.getcwd(), overlap = 0, name_function = 'udf_watershed', conversion_vector = True):
    segmentation_cube = segmentation_cube.apply_neighborhood(
        lambda data: data.run_udf(udf=load_udf(Path(basedir).joinpath(f'{name_function}.py')), runtime='Python'),
        size=[
            {'dimension': 'x', 'value': size, 'unit': 'px'},
            {'dimension': 'y', 'value': size, 'unit': 'px'}
        ],
        overlap=[
            {'dimension': 'x', 'value': overlap, 'unit': 'px'},
            {'dimension': 'y', 'value': overlap, 'unit': 'px'}
        ]
    )

    # vectorization
    if conversion_vector:
        vectorization = segmentation_cube.raster_to_vector()
    else:
        vectorization = segmentation_cube

    return vectorization