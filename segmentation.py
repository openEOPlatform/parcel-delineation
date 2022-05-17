'''
Created on Jun 9, 2020

@author: banyait
'''
import functools
import os
from tensorflow.keras.models import load_model
import numpy as np
import gc
import logging
import itertools
from tensorflow.keras.backend import clear_session
from xarray.core.common import zeros_like, ones_like
from xarray.ufuncs import isnan as ufuncs_isnan
import xarray
from openeo.udf import XarrayDataCube
from typing import Dict
#from tensorflow_core.config import set_visible_devices

# needed because joblib hijacks root logger
logging.basicConfig(level=logging.INFO)


@functools.lru_cache(maxsize=25)
def load_models( modeldir):
    """
    Load the models and make the prediction functions.
    The lru_cache avoids loading the model multiple times on the same worker.

    @param modeldir:
    @return:
    """

    print('Loading convolutional neural networks ...')
    weightsmodel1 = os.path.join(modeldir, 'BelgiumCropMap_unet_3BandsGenerator_Network1.h5')
    weightsmodel2 = os.path.join(modeldir, 'BelgiumCropMap_unet_3BandsGenerator_Network2.h5')
    weightsmodel3 = os.path.join(modeldir, 'BelgiumCropMap_unet_3BandsGenerator_Network3.h5')
    return [
        load_model(weightsmodel1),
        load_model(weightsmodel2),
        load_model(weightsmodel3)
    ]

class Segmentation():
    
    
    def __init__(self, logger=None):
        if logger is None:
            self.log = logging.getLogger(__name__)
        else: self.log=logger
        self.models=None
        #set_visible_devices([], 'GPU')
    
    ################ FUNCTIONS ############################

    def _preprocessing_prediction_modified(self,ndvi_stack, prediction, model, idx_start_filling
                                           , debug = False, patch_size = 128):
        ranges_selection_dates = [int(np.round(item)) for item in
                                  np.arange(0, ndvi_stack.shape[-1], ndvi_stack.shape[-1] / 3)] + [ndvi_stack.shape[-1]]


        ### set zero values to nan temporarily to avoid including
        # them during the statistical calculation
        ndvi_stack[ndvi_stack == 0] = np.nan

        if debug:
            #ndvi_stack = ndvi_stack[128:256,640:768,:]
            import matplotlib.pyplot as plt
        for i in range(3):
            inputs = []
            for s in range(3):
                if i == 0:
                    median_period = np.nanmedian(ndvi_stack[:, :, ranges_selection_dates[s]:ranges_selection_dates[s + 1]], axis=2)
                    # set nan back to zero if present
                    median_period[median_period == np.nan] = 0
                    inputs.append(median_period)
                elif i == 1:
                    max_period = np.nanmax(ndvi_stack[:, :, ranges_selection_dates[s]:ranges_selection_dates[s + 1]], axis=2)
                    max_period[max_period == np.nan] = 0
                    inputs.append(max_period)
                elif i == 2:
                    # mean_period = np.nanmean(ndvi_stack[:, :, ranges_selection_dates[s]:ranges_selection_dates[s + 1]], axis=2)
                    # mean_period[mean_period == np.nan] = 0
                    # inputs.append(mean_period)

                    maxmedian_diff = np.nanmax(ndvi_stack[:, :, ranges_selection_dates[s]:ranges_selection_dates[s + 1]], axis=2)\
                                     -np.nanmedian(ndvi_stack[:, :, ranges_selection_dates[s]:ranges_selection_dates[s + 1]], axis=2)
                    maxmedian_diff[maxmedian_diff ==np.nan] = 0
                    inputs.append(maxmedian_diff)
            if debug:
                outdir = r'/data/users/Private/bontek/git/e-shape/Cases/Ukraine/field delineation/Stratified_test_100_points/Results/openeo/test/TERRA'
                counter = 0
                for input in inputs:
                    plt.imshow(np.transpose(input))
                    if i == 0:
                        plt.savefig(os.path.join(outdir, f'median_test_{str(counter)}.png' ))
                    elif i == 1:
                        plt.savefig(os.path.join(outdir, f'max_test_{str(counter)}.png' ))
                    elif i == 2:
                        plt.savefig(os.path.join(outdir, f'maxmediandiff_test_{str(counter)}.png' ))
                    counter +=1
                    plt.close()



            inputs = np.moveaxis(np.dstack(tuple(inputs)), -1, 0)
            inputs = inputs.transpose([1, 2, 0])
            prediction[:, :, i + idx_start_filling] = np.squeeze(
                model.predict(inputs.reshape(1, inputs.shape[0] * inputs.shape[1], 3)).reshape((patch_size, patch_size)))
        if debug:
            for prediction_nr in range(3):
                plt.imshow(np.transpose(prediction[:,:,prediction_nr +idx_start_filling]))
                plt.savefig(os.path.join(outdir, f'prediction_{str(prediction_nr +idx_start_filling)}.png'))
        return prediction


    #TODO move masking out
    def processWindow(self, models, window, data, modified_prediction = False, debug = False, patch_size = 128):
    
        model1 = models[0]
        model2 = models[1]
        model3 = models[2]
        
        # Read the data
        ndvi_stack = data['s2_ndvi'].values.copy()
        mask_stack = data['s2_mask'].values.copy()
    
        # Mask the ndvi data
        ndvi_stack[mask_stack == 1] = 255
        ndvi_stack[ndvi_stack > 250] = 255
    
        # Count the amount of invalid data per acquisition and sort accordingly
        sum_invalid = np.sum(ndvi_stack == 255, axis=(0, 1))
    
        # if we have enough clear images, we're good to go.
        if not modified_prediction:
            if len(np.where(sum_invalid == 0)[0]) > 3:
                allgood = 1
                self.log.debug((f'Found {len(np.where(sum_invalid == 0)[0])} clear acquisitions -> good to go'))
                ndvi_stack = ndvi_stack[:, :, np.where(sum_invalid == 0)[0]]

            # else we need to add some bad images
            else:
                self.log.debug((f'Found {len(np.where(sum_invalid == 0)[0])} clear acquisitions -> appending some bad images as well!'))
                allgood = 0
                idxsorted = np.argsort(sum_invalid)
                ndvi_stack = ndvi_stack[:, :, idxsorted[0:4]]
        else:
            allgood = 1
    
        # Fill the NaN values
        ndvi_stack[ndvi_stack == 255] = 0
    
        # To fractional number
        ndvi_stack = ndvi_stack / 250.
    
        # Now make sure we can run this window
        nrValidBands = ndvi_stack.shape[2]

        # If the stack hasn't at least 3 bands, we cannot process this window
        if nrValidBands < 3:
            self.log.warning('Not enough input data for window {} -> skipping!'.format(str(window)))
            clear_session()
            #del model1, model2, model3 # TODO only when spark
            gc.collect()
            return None
        else:
            # We'll do 12 predictions: use 3 networks, and for each randomly take 3 NDVI bands and repeat 4 times
            if not modified_prediction:
                prediction = np.zeros((patch_size, patch_size, 12))
            else:
                prediction = np.zeros((patch_size, patch_size, 9))
            if not modified_prediction:
                for i in range(4):
                    prediction[:, :, i] = np.squeeze(
                        model1.predict(ndvi_stack[:, :, np.random.choice(np.arange(nrValidBands), size=3, replace=False)]
                                       .reshape(1, patch_size * patch_size, 3)).reshape((patch_size, patch_size)))

                for i in range(4):
                    prediction[:, :, i + 4] = np.squeeze(
                        model2.predict(
                            ndvi_stack[:, :, np.random.choice(np.arange(nrValidBands), size=3, replace=False)]
                            .reshape(1, patch_size * patch_size, 3)).reshape((patch_size, patch_size)))

                for i in range(4):
                    prediction[:, :, i + 8] = np.squeeze(
                        model3.predict(
                            ndvi_stack[:, :, np.random.choice(np.arange(nrValidBands), size=3, replace=False)]
                            .reshape(1, patch_size * patch_size, 3)).reshape((patch_size, patch_size)))
            else:
                prediction = self._preprocessing_prediction_modified(ndvi_stack, prediction, model1, idx_start_filling=0, debug = debug)
                prediction = self._preprocessing_prediction_modified(ndvi_stack, prediction, model2, idx_start_filling=3,debug = debug)
                prediction = self._preprocessing_prediction_modified(ndvi_stack, prediction, model3, idx_start_filling=6,debug = debug)

            # Final prediction is the median of all predictions per pixel
            clear_session() # IMPORTANT TO AVOID MEMORY LEAK ON THE EXECUTORS!!!
            # del model1, model2, model3 # TODO only when spark
            gc.collect()

            ##now calculate the final prediction value
            ## first check if the variation in
            # prediction is not too big
            prob_perc_25 = np.nanpercentile(prediction, axis = 2, q = [25])[0,:,:]
            prob_perc_75 = np.nanpercentile(prediction, axis = 2, q = [75])[0,:,:]

            range_prob_pixels = prob_perc_75 - prob_perc_25
            range_min_max = np.max(prediction, axis = 2) - np.min(prediction, axis = 2)
            std_prob_pixels = np.std(prediction, axis = 2)
            loc_high_std_pixels = np.where(std_prob_pixels > 0.1)
            loc_high_variation = np.where(range_prob_pixels > 0.6)
            loc_min_max = np.where(range_min_max > 0.3)
            final_prediction = np.median(prediction, axis = 2)
            #final_prediction[loc_high_variation] = prob_perc_25[loc_high_variation]
            #final_prediction[loc_high_std_pixels] = std_prob_pixels[loc_high_std_pixels]
            #final_prediction[loc_min_max] = np.min(prediction, axis = 2)[loc_min_max]

            return window, (final_prediction, allgood)
    


def apply_datacube(cube: XarrayDataCube, context: Dict) -> XarrayDataCube:

    modeldir='/data/users/Public/driesseb/fielddelineation'
    if context is not None:
        modeldir=context.get('modeldir',modeldir)
        modified_prediction = context.get("modified_prediction", False)
        debug = context.get('debug', False)
        patch_size = context.get('patch_size', 128)
    else:
        modified_prediction = False
        debug = False
        patch_size = 0


    cubearray:xarray.DataArray = cube.get_array()
    cubearray = cubearray.transpose('x','bands','y','t')

    nan_locations_NDVI = ufuncs_isnan(cubearray)
    ### clamp out of range NDVI values
    cubearray = cubearray.where(cubearray < 0.92, 0.92)
    cubearray = cubearray.where(cubearray > -0.08,np.nan)

    #set back the original nan values
    cubearray = cubearray.where(~nan_locations_NDVI,np.nan)

    cubearray = ((cubearray +0.08)*250.)[:,0,:,:]

    #cubearray:xarray.DataArray = ((cube.get_array()+0.08)*250.)[:,0,:,:]

    inputarray=cubearray.transpose('x','y','t')
    inputmask=ones_like(inputarray)
    inputmask=inputmask.where(ufuncs_isnan(inputarray), 0.)
    inputdata={
        's2_ndvi': inputarray,
        's2_mask': inputmask
    }

    s=Segmentation()

    models = load_models(modeldir)
    window, (result, allgood) = s.processWindow(models, [], inputdata, modified_prediction = modified_prediction
                                                , debug = debug)

    result=result.astype(np.float64)
    result_xarray = xarray.DataArray(result,dims=['x','y'],coords=dict(x=cubearray.coords['x'],y=cubearray.coords['y']))
    result_xarray=result_xarray.expand_dims('t',0).assign_coords(t=[np.datetime64(str(cubearray.t.dt.year.values[0])+'-01-01')])

    #openEO assumes a fixed data order
    result_xarray = result_xarray.transpose('t','y','x')
    return XarrayDataCube(result_xarray)
    
    
        
    
    
