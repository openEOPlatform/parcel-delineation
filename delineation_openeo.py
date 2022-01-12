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
import utils
import scipy.signal
from pathlib import Path
from print_geojson import print_geojson

openeo_url='http://openeo-dev.vito.be'

centerpoint=[5.0551,51.2182]
year=2019
layerID="TERRASCOPE_S2_TOC_V2"
startdate=str(year)+'-01-01' #'-05-07' #'-08-20'
enddate=str(year)+'-09-30' #'-05-17' #'-09-04'

job_options={
    'driver-memory':'8G',
    'executor-memory':'8G'
}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

utilspath=os.path.dirname(os.path.relpath(utils.__file__,'.'))+'/'

def getImageCollection(eoconn, layer, bbox, bands):
    return eoconn.load_collection(
        layer,
        temporal_extent=[startdate, enddate],
        spatial_extent=dict(zip(["west", "south", "east", "north"], bbox)),
        bands=bands
    ).filter_bbox(crs="EPSG:4326", **dict(zip(["west", "south", "east", "north"], bbox)))

def makekernel(size: int) -> numpy.ndarray:
    assert size % 2 == 1
    kernel_vect = scipy.signal.windows.gaussian(size, std=size / 3.0, sym=True)
    kernel = numpy.outer(kernel_vect, kernel_vect)
    kernel = kernel / kernel.sum()
    return kernel

def create_advanced_mask(band, band_math_workaround=True):
    # in openEO, 1 means mask (remove pixel) 0 means keep pixel
    classification=band
    # keep useful pixels, so set to 1 (remove) if smaller than threshold
    first_mask = ~ ((classification == 4) | (classification == 5) | (classification == 6) | (classification == 7))
    first_mask = first_mask.apply_kernel(makekernel(17))
    # remove pixels smaller than threshold, so pixels with a lot of neighbouring good pixels are retained?
    if band_math_workaround:
        first_mask = first_mask.add_dimension("bands", "mask", type="bands").band("mask")
    first_mask = first_mask > 0.057

    # remove cloud pixels so set to 1 (remove) if larger than threshold
    second_mask = (classification == 3) | (classification == 8) | (classification == 9) | (classification == 10)
    second_mask = second_mask.apply_kernel(makekernel(161))
    if band_math_workaround:
        second_mask = second_mask.add_dimension("bands", "mask", type="bands").band("mask")
    second_mask = second_mask > 0.1
 
    # TODO: the use of filter_temporal is a trick to make cube merging work, needs to be fixed in openeo client
    return first_mask.filter_temporal(startdate, enddate) | second_mask.filter_temporal(startdate, enddate)
    #return first_mask | second_mask
    #return first_mask


def get_resource(relative_path):
    return str(Path( relative_path))

def load_udf(relative_path):
    with open(get_resource(relative_path), 'r+') as f:
        return f.read()
    
def wrap_udf(udffile):
    ws='    '
    header= '# -*- coding: utf-8 -*-\n'+\
            '# Uncomment the import only for coding support~\n'+\
            'from openeo_udf.api.datacube import DataCube\n'+\
            'from typing import Dict\n'+\
            'def apply_hypercube(cube: DataCube, context: Dict) -> DataCube:\n'
    footer= ws+'\n'+ws+'return inner_apply_hypercube(cube,context)\n'
    udf=header+ws+load_udf(udffile).replace('\n','\n'+ws)+footer
    return udf

def replaceparams_udf(udf,params=None):
    if params is not None:
        p=json.dumps(params)+'\n'
        udf=udf.replace("udfparams={}","udfparams="+p)
    return udf


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
    #latlon = pyproj.Proj('EPSG:4326')
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
    
    return bboxes

def computebboxgeojson(bboxmatrix):
    geojson={
        "type":"FeatureCollection",
        "name": "bboxaround_{}:{}".format(*centerpoint),
#        "crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},
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
    eoconn.authenticate_oidc()#basic(openeo_user,openeo_pass)
    
    #build the bounding boxes
    bboxes=computebboxmatrix(centerpoint, 1)
    bboxgeojson=computebboxgeojson(bboxes)

    # compute the mask
    maskband=create_advanced_mask(getImageCollection(eoconn, layerID, bboxes[0,0,:], ["SCENECLASSIFICATION_20M"]).band("SCENECLASSIFICATION_20M"))\

    # compute ndvi
    ndviband=getImageCollection(eoconn, layerID, bboxes[0,0,:], ["TOC-B04_10M","TOC-B08_10M"])\
        .ndvi(red="TOC-B04_10M",nir="TOC-B08_10M")

    # set NaN where mask is active
    ndviband=ndviband.mask(maskband)

    # select top 3 usable layers
    ndviband=ndviband.apply_dimension(load_udf('udf_reduce_images.py'),dimension='t',runtime="Python")
    
    # produce the segmentation image
    segmentationband=ndviband.apply_dimension(wrap_udf('segmentation_core.py'), dimension='t', runtime="Python")

    # postprocess for vectorization
    segmentationband=segmentationband.apply_dimension(load_udf('udf_sobel_felzenszwalb.py'), dimension='t', runtime="Python")

    # vectorization
    vectorization=segmentationband.raster_to_vector()

    with open('delineation_process','w') as f: json.dump(vectorization.graph, f, indent=2) 

    job = vectorization.execute_batch("result_vectorization.json",job_options=job_options)
    job.get_results().download_file("results/result_vectorized.json")

    print_geojson('results/result_vectorization.json', 'results/result_vectorization_corrected.json')
            
    logger.info('FINISHED')






