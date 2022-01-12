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

centerpoint=[5.0551,51.2182]
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

def computebboxmatrix(centerpoint,levels):
    size=1000. #2560. # openeo wrks with 256x256 blocks and the resolution is 10m
    overlap=320. # the underlying neural network uses 32 pixel borders on an 10m resolution image
    epsilon=1. # 
    
    # get centerpoint meters
    p = pyproj.Proj(proj='utm',zone=zone(centerpoint),ellps='WGS84',preserve_units=False)
    x,y = p(*centerpoint)    

    # build the meters offset structure
    bboxes=numpy.zeros((2*levels-1,2*levels-1,4),dtype=numpy.float64)
    for i in range(2*levels-1):
        bboxes[i,:,0]=(i-(levels-0.5))*size-(i-(levels-1.))*overlap
        bboxes[:,i,1]=(i-(levels-0.5))*size-(i-(levels-1.))*overlap
    bboxes[:,:,2]=bboxes[:,:,0]+size
    bboxes[:,:,3]=bboxes[:,:,1]+size
    
    # add the x,y coordinates
    bboxes[:,:,0]=bboxes[:,:,0]+x+epsilon
    bboxes[:,:,1]=bboxes[:,:,1]+y+epsilon
    bboxes[:,:,2]=bboxes[:,:,2]+x-epsilon
    bboxes[:,:,3]=bboxes[:,:,3]+y-epsilon

    logger.info("UTM ZONE: "+str(zone(centerpoint)))
    
    # convert back to lat/lon
    for i in range(2*levels-1):
        for j in range(2*levels-1):
            bboxes[i,j,0],bboxes[i,j,1]=p(bboxes[i,j,0],bboxes[i,j,1],inverse=True)
            bboxes[i,j,2],bboxes[i,j,3]=p(bboxes[i,j,2],bboxes[i,j,3],inverse=True)
    
    return bboxes[0, 0, :]

def computebboxgeojson(bboxmatrix):
    geojson={
        "type":"FeatureCollection",
        "name": "bboxaround_{}:{}".format(*centerpoint),
        "features":[]
    }
    for i in range(bboxmatrix.shape[0]):
        for j in range(bboxmatrix.shape[1]):
            b=bboxmatrix[i,j]
            geojson["features"].append({
                "type":"Feature",
                "properties":{ "xtile":i, "ytile":j },
                "geometry":{"type":"Polygon","coordinates":[[ [b[0],b[1]], [b[0],b[3]], [b[2],b[3]], [b[2],b[1]], [b[0],b[1]] ]]}
            })
    
    print(json.dumps(geojson))
    return geojson


    
if __name__ == '__main__':
    
    # connection
    eoconn=openeo.connect(openeo_url)
    eoconn.authenticate_oidc()
    
    #build the bounding boxes
    bbox=computebboxmatrix(centerpoint, 1)

    s2_bands = eoconn.load_collection(
        "TERRASCOPE_S2_TOC_V2",
        temporal_extent=[startdate, enddate],
        spatial_extent=dict(zip(["west", "south", "east", "north"], bbox)),
        bands=["B04", "B08","SCL"]
    )
    s2_bands = s2_bands.process("mask_scl_dilation", data=s2_bands, scl_band_name="SCL")

    ndviband= s2_bands.ndvi(red="B04", nir="B08")

    #ndviband.download("all_inputs.nc")

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
    segmentationband=segmentationband.apply_dimension(load_udf('udf_sobel_felzenszwalb.py'), dimension='t', runtime="Python")

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






