# -*- coding: utf-8 -*-
# Uncomment the import only for coding support
from openeo.udf import XarrayDataCube
from typing import Dict

def convert_nc_tiff(xr_ds, array, outname, data_type = 'float64'):
    import osr
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window
    epsg = 32635

    height = array.shape[0]
    width = array.shape[1]
    transform = from_origin(min(xr_ds['x'].values), max(xr_ds['y'].values), 10, 10)
    profile = {'driver': 'GTiff', 'height': height, 'width': width, 'count': 1, 'dtype': data_type,
               'transform': transform, 'crs': 'EPSG:{}'.format(epsg)}

    config = {}
    config['output_dataset'] = rasterio.open(outname, 'w', **profile)
    config['output_dataset'].scales = [0]
    config['output_dataset'].offset = [0]
    config['output_dataset'].types = [data_type]
    config['output_dataset'].write(array, 1, window=Window(0, 0, array.shape[1], array.shape[0]))
    config['output_dataset'].close()

def apply_datacube(cube: XarrayDataCube, context: Dict) -> XarrayDataCube:
    import numpy as np
    from skimage import segmentation
    from skimage.filters import sobel
    from skimage.future import graph
    import xarray

    # get the underlying numpy array
    inarray=cube.get_array().squeeze('t',drop=True).squeeze('bands',drop=True)
    inimage=inarray.values
    # inimage-=np.min(inimage)
    # inimage=inimage*249./np.max(inimage)
    # image=np.clip((inimage-0.3*250)*2,0.,249.)

    # compute edges
    edges=sobel(inimage)

    # # convert to tiff
    # convert_nc_tiff(inarray, edges,
    #                 r'/data/users/Private/bontek/git/e-shape/Cases/Ukraine/field delineation/Stratified_test_100_points/Results/openeo/test/TERRA/edges.tiff')


    # Perform felzenszwalb segmentation
    segment = np.array(segmentation.felzenszwalb(inimage, scale=120, sigma=0., min_size=30, multichannel=False)).astype(np.int32)

    # convert to tiff
    # convert_nc_tiff(inarray, segment,
    #                 r'/data/users/Private/bontek/git/e-shape/Cases/Ukraine/field delineation/Stratified_test_100_points/Results/openeo/test/TERRA/felzenswalb_segment_scale_120.tiff',
    #                 data_type='int32')

    # Perform the rag boundary analysis and merge the segments
    bgraph = graph.rag_boundary(segment, edges)
    # merging segments
    mergedsegment = graph.cut_threshold(segment, bgraph, 0.15, in_place=False)

    # # convert to tiff
    # convert_nc_tiff(inarray, mergedsegment,
    #                 r'/data/users/Private/bontek/git/e-shape/Cases/Ukraine/field delineation/Stratified_test_100_points/Results/openeo/test/TERRA/merged_segm_020_thr.tiff',
    #                 data_type='int32')

    # segments start from 0, therefore the 0th has to be moved
    #mergedsegment[mergedsegment==0]=np.max(mergedsegment)+1 # -> Seems to be not needed ?!
    # We currently take 0.3 as the binary threshold to distinguish between segments of fields and other segments.
    # This could definitely be improved and made more objective.
    # NOTE: new implementation uses scaled data, so threshold needs to be scaled as well!

    ### create random numbers for the segments
    unique_classes = np.unique(mergedsegment)
    random_numbers = np.random.randint(0, 1000000, size = len(np.unique(mergedsegment)))

    counter = 0
    for unique_class in unique_classes:
        if unique_class == 0:
            continue
        mergedsegment[mergedsegment == unique_class] = random_numbers[counter]
        counter += 1


    mergedsegment = mergedsegment.astype(float)
    mergedsegment[inimage < 0.3] = np.nan
    mergedsegment[mergedsegment < 0] = 0

    # # convert to tiff
    # convert_nc_tiff(inarray, mergedsegment,
    #                 r'/data/users/Private/bontek/git/e-shape/Cases/Ukraine/field delineation/Stratified_test_100_points/Results/openeo/test/SHUB/merged_segm_015_thr_cleaned_scale_120.tiff'
    #                 )



    outarr=xarray.DataArray(mergedsegment.reshape(cube.get_array().shape),dims=cube.get_array().dims,coords=cube.get_array().coords)
    outarr=outarr.astype(np.float64)
    outarr=outarr.where(outarr!=0,np.nan)

    return XarrayDataCube(outarr)
