# -*- coding: utf-8 -*-
# Uncomment the import only for coding support
from openeo.udf import XarrayDataCube
from typing import Dict


def apply_datacube(cube: XarrayDataCube, context: Dict) -> XarrayDataCube:

    import numpy

    # set how much images to select in the order of highest number of clear pixels
    maxlayers=12
    if context is not None:
        maxlayers=context.get('maxlayers',maxlayers)
        modified_filter = context.get("modified_filter", False)
        debug = context.get('debug', False)
    else:
        debug = False
        modified_filter = False

    # get the underlying xarray
    inputarray=cube.get_array()

    #check if not 4D:
    if debug:
        if len(inputarray.shape) == 4:
            #turn to 3D
            inputarray = inputarray[0]

    # prepare uniform coordinates
    trange=numpy.arange(numpy.datetime64(str(inputarray.t.dt.year.values[0])+'-01-01'),numpy.datetime64(str(inputarray.t.dt.year.values[0])+'-03-31'))

    # order the layers by decreasing number of clear pixels
    counts=list(sorted(zip(
        [i for i in range(inputarray.t.shape[0])],
        inputarray.count(dim=['x','y']).values.flatten()
    ), key=lambda i: i[1], reverse=True))
    if modified_filter:
        cloud_thr = numpy.arange(10,25,5)
        clearest_image = max([i[1] for i in counts]) #border are set to nan so not possible tpo calculate percentage of nan
        nr_valid_indices = 0
        counter = 0
        while nr_valid_indices <4:
            counts_kept = [i for i in counts if (1 - (i[1] / clearest_image)) * 100 < cloud_thr[counter]]
            nr_valid_indices = len(counts_kept)
            counter += 1
            if counter == len(cloud_thr):
                break
        if nr_valid_indices >3:
            counts = counts_kept

        resultarray=inputarray[[i[0] for i in counts]]
        resultarray=resultarray.sortby(resultarray.t,ascending=True)
        resultarray=resultarray.assign_coords(t=trange[:resultarray.t.values.size])

    else:
        # return the selected ones
        resultarray=inputarray[[i[0] for i in counts[:maxlayers]]]
        resultarray=resultarray.sortby(resultarray.t,ascending=True)
        resultarray=resultarray.assign_coords(t=trange[:maxlayers])
    return XarrayDataCube(resultarray)

