'''
Created on Jun 9, 2020

@author: banyait
'''

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
#from tensorflow_core.config import set_visible_devices

# needed because joblib hijacks root logger
logging.basicConfig(level=logging.INFO)

class Segmentation():
    
    
    def __init__(self, logger=None):
        if logger is None:
            self.log = logging.getLogger(__name__)
        else: self.log=logger
        self.models=None
        #set_visible_devices([], 'GPU')
    
    ################ FUNCTIONS ############################
    
    def load_models(self,modeldir):
        # Load the models and make the prediction functions
        if self.models is None: 
            self.log.info('Loading convolutional neural networks ...')
            weightsmodel1 = os.path.join(modeldir, 'BelgiumCropMap_unet_3BandsGenerator_Network1.h5')
            weightsmodel2 = os.path.join(modeldir, 'BelgiumCropMap_unet_3BandsGenerator_Network2.h5')
            weightsmodel3 = os.path.join(modeldir, 'BelgiumCropMap_unet_3BandsGenerator_Network3.h5')
            self.models = [
                load_model(weightsmodel1),
                load_model(weightsmodel2),
                load_model(weightsmodel3)
            ]
        else: self.log.info('Reusing convolutional neural networks ...')
        return self.models
    
    #TODO move masking out
    def processWindow(self, models, window, data):
    
        model1 = models[0]
        model2 = models[1]
        model3 = models[2]
    
        #from parcel.utils.raster import Window
        #windowObj = Window(window=window, tile=tileobj)
    
        # Read the data
        #ndvi_stack = windowObj.get_raw_arrays('s2_ndvi')
        #mask_stack = windowObj.get_raw_arrays('s2_mask')
        ndvi_stack = data['s2_ndvi'].values.copy()
        mask_stack = data['s2_mask'].values.copy()
    
        # Mask the ndvi data
        ndvi_stack[mask_stack == 1] = 255
        ndvi_stack[ndvi_stack > 250] = 255
    
        # Count the amount of invalid data per acquisition and sort accordingly
        sum_invalid = np.sum(ndvi_stack == 255, axis=(0, 1))
    
        # if we have enough clear images, we're good to go.
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
            prediction = np.zeros((128, 128, 12))
            for i in range(4):
                prediction[:, :, i] = np.squeeze(
                    model1.predict(ndvi_stack[:, :, np.random.choice(np.arange(nrValidBands), size=3, replace=False)]
                                   .reshape(1, 128 * 128, 3)).reshape((128, 128)))
            for i in range(4):
                prediction[:, :, i + 4] = np.squeeze(
                    model2.predict(ndvi_stack[:, :, np.random.choice(np.arange(nrValidBands), size=3, replace=False)]
                                   .reshape(1, 128 * 128, 3)).reshape((128, 128)))
            for i in range(4):
                prediction[:, :, i + 8] = np.squeeze(
                    model3.predict(ndvi_stack[:, :, np.random.choice(np.arange(nrValidBands), size=3, replace=False)]
                                   .reshape(1, 128 * 128, 3)).reshape((128, 128)))
    
    
            # Final prediction is the median of all predictions per pixel
            clear_session() # IMPORTANT TO AVOID MEMORY LEAK ON THE EXECUTORS!!!
            # del model1, model2, model3 # TODO only when spark
            gc.collect()
            return window, (np.median(prediction, axis=2), allgood)
    
    
        
    def processWindowList(self,windowlist,modeldir,inputdata,stride):
        bbox=(
            (min([i[0][0] for i in windowlist]), max([i[0][1] for i in windowlist])),
            (min([i[1][0] for i in windowlist]), max([i[1][1] for i in windowlist]))
    #             (windowlist[0][0][0],windowlist[len(windowlist)-1][0][1]) , 
    #             (windowlist[0][1][0],windowlist[len(windowlist)-1][1][1]) 
        )
        #retvals=[]
        result=zeros_like(inputdata['s2_ndvi'][:,:,0]).astype(np.ubyte)
        models=self.load_models(modeldir)
        for window in windowlist:
            inputdatawindow={}
            for (k,v) in inputdata.items(): inputdatawindow[k]=v[
                window[0][0]-bbox[0][0]:window[0][1]-bbox[0][0],
                window[1][0]-bbox[1][0]:window[1][1]-bbox[1][0]
            ]
            window, (winresult, allgood) = self.processWindow(models, window, inputdatawindow)
            if winresult is not None:
                # Scale to byte
                data = winresult # winresult[stride:-stride, stride:-stride]
                data = data * 250
                data = data.astype(np.ubyte) # TODO move this out to file-based
                # when a window is on the side of the list, then including it
                #  - center 64x64 windows always come from the 128x128 centers 
                #  - bbox border is filled randomly as the windows were processed
                sxmin= stride if window[0][0]!=bbox[0][0] else 0
                sxmax=-stride if window[0][1]!=bbox[0][1] else 0
                symin= stride if window[1][0]!=bbox[1][0] else 0
                symax=-stride if window[1][1]!=bbox[1][1] else 0
                # write window to destination
                subWindow = (
    #                     (window[0][0] + stride, window[0][1] - stride),
    #                     (window[1][0] + stride, window[1][1] - stride)
                    (window[0][0]+sxmin, window[0][1]+sxmax),
                    (window[1][0]+symin, window[1][1]+symax)
                )
                # We had to pull in some bad images as well, need to log this!
                if allgood == 0: self.log.debug("Window {} successfully processed, but some bad images were included!".format(window))
                else: self.log.debug("Window {} successfully processed!".format(window))
                #retvals.append((subWindow,data))
                result[
                    subWindow[0][0]-bbox[0][0]:subWindow[0][1]-bbox[0][0],
                    subWindow[1][0]-bbox[1][0]:subWindow[1][1]-bbox[1][0]
                ]=data[
                    0+sxmin:data.shape[0]+sxmax,
                    0+symin:data.shape[1]+symax
                ]
            else:
                self.log.error('Window {} returned an invalid result -> should investigate!'.format(window))
    
        #return retvals
        adjusted_bbox=((bbox[0][0]+stride,bbox[0][1]-stride),(bbox[1][0]+stride,bbox[1][1]-stride))
        return (adjusted_bbox,result)
    
    
    def computeWindowLists(self, bboxWindow, imageSize, windowsize, stride):
        '''
        bboxWindow: ((xmin,xmax),(ymin,ymax)) or None to use full image
        imageSize: (width,height)
        windowSize: size of blocks to split bboxWindow
        stride: overlaps width neighbours
        
        returns: 2d list of windows, where each window element is in the format ((xmin,xmax),(ymin,ymax))
        '''
        if bboxWindow is None:  bbox=[0,0,imageSize[0],imageSize[1]]
        else: bbox=[bboxWindow[0][0],bboxWindow[1][0],bboxWindow[0][1],bboxWindow[1][1]]
        
        # because sride amount of frame is not filled in the wind with windowsize -> bbox has to be enlarged
        bbox[0]= bbox[0]-stride if bbox[0]-stride>=0 else 0 
        bbox[1]= bbox[1]-stride if bbox[1]-stride>=0 else 0
        bbox[2]= bbox[2]+stride if bbox[2]+stride<=imageSize[0] else imageSize[0]
        bbox[3]= bbox[3]+stride if bbox[3]+stride<=imageSize[1] else imageSize[1]
         
        # We need to check if we're at the end of the master image
        # We have to make sure we have a full subtile
        # so we need to expand such tile and the resulting overlap
        # with previous subtile is not an issue
        windowlist=[]
        for xStart in range(bbox[0], bbox[2], windowsize - 2 * stride):
            
            windowlist.append([])
            
            if xStart + windowsize > bbox[2]:
                xStart = bbox[2] - windowsize
                xEnd = bbox[2]
            else:
                xEnd = xStart + windowsize
    
            for yStart in range(bbox[1], bbox[3], windowsize - 2 * stride):
                if yStart + windowsize > bbox[3]:
                    yStart = bbox[3] - windowsize
                    yEnd = bbox[3]
                else:
                    yEnd = yStart + windowsize
    
                windowlist[len(windowlist)-1].append(((xStart, xEnd), (yStart, yEnd)))
        
                if (yEnd==bbox[3]): break
            if (xEnd==bbox[2]): break
    
        return windowlist
    
# openeo runner
def inner_apply_hypercube(cube,context):

    from openeo_udf.api.datacube import DataCube

    modeldir='/data/users/Public/driesseb/fielddelineation'
    if context is not None:
        modeldir=context.get('modeldir',modeldir)
    
    cubearray:xarray.DataArray = ((cube.get_array()+0.08)*250.)[:,0,:,:]
    inputarray=cubearray.transpose('x','y','t')
    inputmask=ones_like(inputarray)
    inputmask=inputmask.where(ufuncs_isnan(inputarray), 0.)
    inputdata={
        's2_ndvi': inputarray,
        's2_mask': inputmask
    }

    s=Segmentation()
    windows=s.computeWindowLists(None, inputarray.shape[0:2], windowsize=128, stride=32)
    windowlist=list(itertools.chain.from_iterable(windows))

    result=s.processWindowList(
        windowlist, 
        modeldir, 
        inputdata, 
        32
    )[1]
    
    result=result.astype(np.float64)
    result=result.expand_dims('bands',0).assign_coords(bands=['delineation'])
    result=result.expand_dims('t',0).assign_coords(t=[np.datetime64(str(cubearray.t.dt.year.values[0])+'-01-01')])
#    result=result.astype(numpy.float64)    
    return DataCube(result)
    
    
        
    
    
