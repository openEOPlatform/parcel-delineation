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

openeo_url='openeo-dev.vito.be'

bbox = [5.04776065, 51.213841, 5.06244073, 51.22255853]
year=2019
layerID="TERRASCOPE_S2_TOC_V2"
layerID_sentinelhub="SENTINEL2_L2A"

startdate=str(year)+'-01-01'
enddate=str(year)+'-09-30'

job_options={
    'driver-memory':'8G',
    'executor-memory':'8G'
}

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

    eoconn=openeo.connect(openeo_url)
    eoconn.authenticate_oidc()

    s2_bands = eoconn.load_collection(
        "TERRASCOPE_S2_TOC_V2",
        temporal_extent=[startdate, enddate],
        spatial_extent=dict(zip(["west", "south", "east", "north"], bbox)),
        bands=["B04", "B08","SCL"]
    )
    s2_bands = s2_bands.process("mask_scl_dilation", data=s2_bands, scl_band_name="SCL")

    ndviband= s2_bands.ndvi(red="B04", nir="B08")

    # select top 12 usable layers
    ndviband=ndviband.apply_dimension(load_udf('udf_reduce_images.py'),dimension='t',runtime="Python")

    #ndviband.download("ndvi_inputs.nc")
    
    # produce the segmentation image
    segmentationband = ndviband.apply_neighborhood(
        lambda data: data.run_udf(udf=load_udf('segmentation.py'), runtime='Python',
                                  context={
                                      'startdate': startdate,
                                      'enddate': enddate
                                  }),
        size=[
            {'dimension': 'x', 'value': 64, 'unit': 'px'},
            {'dimension': 'y', 'value': 64, 'unit': 'px'}
        ],
        overlap=[
            {'dimension': 'x', 'value': 32, 'unit': 'px'},
            {'dimension': 'y', 'value': 32, 'unit': 'px'}
        ]
    )
    #segmentationband.download("segmented.tiff")

    # postprocess for vectorization
    segmentationband=segmentationband.apply_neighborhood(
        lambda data: data.run_udf(udf=load_udf('udf_sobel_felzenszwalb.py'), runtime='Python'),
        size=[
            {'dimension': 'x', 'value': 512, 'unit': 'px'},
            {'dimension': 'y', 'value': 512, 'unit': 'px'}
        ],
        overlap=[
            {'dimension': 'x', 'value': 0, 'unit': 'px'},
            {'dimension': 'y', 'value': 0, 'unit': 'px'}
        ]
    )


    # vectorization
    vectorization=segmentationband.raster_to_vector()

    result = vectorization.download("results/out.json")

    with open("results/out.json") as f:
        polygons = json.load(f)
    from shapely.geometry import shape
    import geopandas as gpd
    geom = [shape(p) for p in polygons]
    gpd.GeoDataFrame(geometry=geom,crs="EPSG:32631").to_file("results/parcels.gpkg", layer='parcels', driver="GPKG")
            
    logger.info('FINISHED')






