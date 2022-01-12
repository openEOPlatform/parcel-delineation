# -*- coding: utf-8 -*-
# Uncomment the import only for coding support
from openeo_udf.api.datacube import DataCube
from typing import Dict


def apply_hypercube(cube: DataCube, context: Dict) -> DataCube:
    # import sys
    # sys.path.append(r'/data/users/Public/driesseb/dep/')
    

    import numpy as np
    from skimage import segmentation
    from skimage.filters import sobel
    from skimage.future import graph
    import xarray

    # get the underlying numpy array
    inarray=cube.get_array().squeeze('t',drop=True).squeeze('bands',drop=True)
    inimage=inarray.values#(inarray.values*inarray.values)/255
    inimage-=np.min(inimage)
    inimage=inimage*249./np.max(inimage)
    image=np.clip((inimage-0.3*250)*2,0.,249.)
#    image[image < 0.3 * 250]=0

    # compute edges
    edges=sobel(image)

    # Perform felzenszwalb segmentation
    segment = np.array(segmentation.felzenszwalb(image, scale=1, sigma=0., min_size=30, multichannel=False)).astype(np.int32)
    # Perform the rag boundary analysis and merge the segments
    bgraph = graph.rag_boundary(segment, edges)
    # merging segments
    mergedsegment = graph.cut_threshold(segment, bgraph, 0.15, in_place=False)
    # segments start from 0, therefore the 0th has to be moved
    mergedsegment[mergedsegment==0]=np.max(mergedsegment)+1
    # We currently take 0.3 as the binary threshold to distinguish between segments of fields and other segments.
    # This could definitely be improved and made more objective.
    # NOTE: new implementation uses scaled data, so threshold needs to be scaled as well!
    mergedsegment[image==0] = 0
    #mergedsegment[image < 0.3 * 250] = 0
    mergedsegment[mergedsegment < 0] = 0
    #mergedsegment[mergedsegment > 0] = 200

    outarr=xarray.DataArray(mergedsegment.reshape(cube.get_array().shape),dims=cube.get_array().dims,coords=cube.get_array().coords)
    outarr=outarr.astype(np.float64)
    outarr=outarr.where(outarr!=0,np.nan)

#     #############################################
#  
#     attent=1.01**(0.5*image)
#     attent-=np.min(attent)
#     attent1=attent*249./np.max(attent)
#  
#     attent=(image-0.3*250)*2
#     attent2=np.clip(attent,0.,249.)
#  
#     from parcel.feature.segmentation.new_tamasmerge.print_xdatacubexarray import print_xarray_dataarray
#     arrs=np.expand_dims(np.concatenate((
#         np.expand_dims(inimage,0),
#         np.expand_dims(image,0),
#         np.expand_dims(mergedsegment,0),
#         np.expand_dims(outarr[0,0].values,0)
#     )),0)
#     xarr=xarray.DataArray(arrs,dims=cube.get_array().dims,coords={'bands':[
#         'input',
#         'edges',
#         'segments',
#         'mergedsegments'
#     ],'t':[np.datetime64('2019-01-01')]})
#     print_xarray_dataarray('sobelfelzenswald',xarr)

    return DataCube(outarr)
