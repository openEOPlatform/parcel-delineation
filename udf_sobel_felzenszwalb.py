# -*- coding: utf-8 -*-
# Uncomment the import only for coding support
from openeo.udf import XarrayDataCube
from typing import Dict


def apply_datacube(cube: XarrayDataCube, context: Dict) -> XarrayDataCube:
    import numpy as np
    from skimage import segmentation
    from skimage.filters import sobel
    from skimage.future import graph
    import xarray

    # get the underlying numpy array
    inarray=cube.get_array().squeeze('t',drop=True).squeeze('bands',drop=True)
    inimage=inarray.values
    inimage-=np.min(inimage)
    inimage=inimage*249./np.max(inimage)
    image=np.clip((inimage-0.3*250)*2,0.,249.)

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
    mergedsegment[mergedsegment < 0] = 0

    outarr=xarray.DataArray(mergedsegment.reshape(cube.get_array().shape),dims=cube.get_array().dims,coords=cube.get_array().coords)
    outarr=outarr.astype(np.float64)
    outarr=outarr.where(outarr!=0,np.nan)

    return XarrayDataCube(outarr)
