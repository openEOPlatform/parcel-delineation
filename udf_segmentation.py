import functools
import gc
import logging
import os
from typing import Dict

import numpy as np
import xarray
from openeo.udf import XarrayDataCube, inspect
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import load_model
from xarray.core.common import ones_like
from xarray.ufuncs import isnan as ufuncs_isnan

_log = logging.getLogger(__name__)


@functools.lru_cache(maxsize=25)
def load_models(modeldir):
    """
    Load the models and make the prediction functions.
    The lru_cache avoids loading the model multiple times on the same worker.

    @param modeldir: Model directory
    @return: Loaded models
    """

    print("Loading convolutional neural networks ...")
    weightsmodel1 = os.path.join(modeldir, "BelgiumCropMap_unet_3BandsGenerator_Network1.h5")
    weightsmodel2 = os.path.join(modeldir, "BelgiumCropMap_unet_3BandsGenerator_Network2.h5")
    weightsmodel3 = os.path.join(modeldir, "BelgiumCropMap_unet_3BandsGenerator_Network3.h5")
    return [
        load_model(weightsmodel1),
        load_model(weightsmodel2),
        load_model(weightsmodel3),
    ]


def processWindow(models, ndvi_stack, patch_size=128):
    ## check whether you actually supplied 3 images or more
    nrValidBands = ndvi_stack.shape[2]

    ## if the stack doesn't have at least 3 bands, we cannot process this window
    if nrValidBands < 3:
        inspect(message="Not enough input data for this window -> skipping!")
        clear_session()
        gc.collect()
        return None
    nbModels = 2
    nbPerModel = 1
    ## we'll do 12 predictions: use 3 networks, and for each randomly take 3 NDVI bands and repeat 4 times
    prediction = np.zeros((patch_size, patch_size, nbModels * nbPerModel))
    for model_counter in range(nbModels):
        for i in range(nbPerModel):
            prediction[:, :, i + nbPerModel*model_counter] = np.squeeze(
                models[model_counter].predict(
                    ndvi_stack[
                        :,
                        :,
                        np.random.choice(
                            np.arange(nrValidBands), size=3, replace=False
                        ),
                    ].reshape(1, patch_size * patch_size, 3)
                ).reshape((patch_size, patch_size))
            )
    clear_session()  # to avoid memory leakage executors
    gc.collect()

    ## final prediction is the median of all predictions per pixel
    final_prediction = np.median(prediction, axis=2)
    return final_prediction


def preprocessDatacube(cubearray: XarrayDataCube) -> XarrayDataCube:
    cubearray = cubearray.transpose("x", "bands", "y", "t")

    ## get nan indices
    nan_locations_NDVI = ufuncs_isnan(cubearray)

    ## clamp out of range NDVI values
    cubearray = cubearray.where(cubearray < 0.92, 0.92)
    cubearray = cubearray.where(cubearray > -0.08, np.nan)

    ## xarray nan to np nan (we will pass a numpy array)
    cubearray = cubearray.where(~nan_locations_NDVI, np.nan)

    ## rescale data and just take NDVI to get rid of the band dimension
    cubearray = ((cubearray + 0.08) * 250.0)[:, 0, :, :]

    ## transpose to format accepted by model and get the values
    ndvi_stack = cubearray.transpose("x", "y", "t").values

    ## create a mask where all valid values are 1 and all nans are 0
    mask_stack = ones_like(cubearray)
    mask_stack = mask_stack.where(ufuncs_isnan(cubearray), 0.0).values

    ## mask the ndvi data
    ndvi_stack[mask_stack == 1] = 255
    ndvi_stack[ndvi_stack > 250] = 255

    ## count the amount of invalid data per acquisition and sort accordingly
    sum_invalid = np.sum(ndvi_stack == 255, axis=(0, 1))

    ## if we have enough clear images (without ANY missing values), we're good to go, 
    ## and we will use all of them (could be more than 3!)
    if len(np.where(sum_invalid == 0)[0]) > 3:
        #inspect(f"Found {len(np.where(sum_invalid == 0)[0])} clear acquisitions -> good to go")
        ndvi_stack = ndvi_stack[:, :, np.where(sum_invalid == 0)[0]]

    ## else we need to add some images that do contain some nan's; in this case we will select just the 3 best ones
    else:
        #inspect(f"Found {len(np.where(sum_invalid == 0)[0])} clear acquisitions -> appending some bad images as well!")
        idxsorted = np.argsort(sum_invalid)
        ndvi_stack = ndvi_stack[:, :, idxsorted[:3]]

    ## fill the NaN values with 0
    ndvi_stack[ndvi_stack == 255] = 0

    ## convert to fractional number
    ndvi_stack = ndvi_stack / 250.0

    ## return the stack
    return ndvi_stack

def apply_datacube(cube: XarrayDataCube, context: Dict) -> XarrayDataCube:
    
    ## load in the pretrained U-Net keras models and do inference!
    modeldir = "/data/users/Public/driesseb/fielddelineation" #TODO: replace this with ONNX
    models = load_models(modeldir)

    ## get the array and transpose
    cubearray: xarray.DataArray = cube.get_array()
    
    ## preprocess the datacube
    ndvi_stack = preprocessDatacube(cubearray)

    ## process the window
    result = processWindow(models, ndvi_stack)

    ## transform your numpy array predictions into an xarray
    result = result.astype(np.float64)
    result_xarray = xarray.DataArray(
        result,
        dims=["x", "y"],
        coords={"x": cubearray.coords["x"], "y": cubearray.coords["y"]},
    )
    # Reintroduce time and bands dimensions
    result_xarray = result_xarray.expand_dims(
        dim={
            "t": [np.datetime64(str(cubearray.t.dt.year.values[0]) + "-01-01")],
            "bands": ["prediction"],
        },
    )
    # Return the result
    return XarrayDataCube(result_xarray)
