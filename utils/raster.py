"""
Created on Tue Dec 06 14:50:02 2016

@author: VTRICHTK
"""

import osr
import gdal
import subprocess
import logging
import pandas as pd
import numpy as np
import os
import glob
import sys
import rasterio
import xarray as xr
import ogr
from rasterio import features

import geopandas as gpd

log = logging.getLogger(__name__)
_GLOBAL_NAMESPACE = dict([(key, getattr(np, key))
                          for key in dir(np) if not key.startswith('__')])


class Vrt:
    '''
    An abstract class that deals with tile-specifc VRTs that are needed for
    various data sources
    '''

    def __init__(self, sourcedir, vrtdir, startdate, enddate, inpattern,
                 infiles, outpattern, dtype=np.float,
                 operation='1*x', nodata=None, dropna=False, mask=None):

        self._sourcedir = sourcedir
        self._startdate = pd.to_datetime(startdate)
        self._enddate = pd.to_datetime(enddate)
        self.dtype = dtype
        self.operation = operation
        self.nodata = nodata
        self.dropna = dropna
        self.mask = mask

        self.file, self.dates, self.sourcefiles = self._create(vrtdir,
                                                               inpattern,
                                                               infiles,
                                                               outpattern)

    def _create(self, vrtdir, inpattern, infiles, outpattern):

        outvrt = os.path.join(vrtdir,
                              outpattern + '_'.join(
                                  [self._startdate.strftime('%Y%m%d'),
                                   self._enddate.strftime('%Y%m%d')]) + '.vrt')

        self.name = outpattern + '_'.join(
            [self._startdate.strftime('%Y%m%d'),
             self._enddate.strftime('%Y%m%d')])

        if os.path.exists(outvrt): os.remove(outvrt)
        log.info('Creating: {}'.format(outvrt))

        if infiles is None:
            img_files = glob.glob(os.path.join(self._sourcedir, inpattern))
        else:
            img_files = infiles

        if len(img_files) == 0:
            log.info('No img files found under: {}'.format(
                os.path.join(self._sourcedir, inpattern)))

        indates = pd.to_datetime(
            np.array([os.path.basename(x).split('_')[1][0:8] for
                      x in img_files]))
        mask = (indates >= self._startdate) & (indates <= self._enddate)

        img_files = np.array(img_files)[mask]
        indates = indates[mask]
        idx_sort = np.argsort(indates)
        img_files = list(img_files[idx_sort])

        # Create a file that will hold the individual files
        filename = os.path.join(vrtdir,
                                '_'.join(['filelist', outpattern,
                                          self._startdate.strftime(
                                              '%Y%m%d'),
                                          self._enddate.strftime(
                                              '%Y%m%d')]) + '.txt')

        if os.path.exists(filename):
            os.remove(filename)
        fileList = open(filename, 'w')

        for file in img_files:
            fileList.write(file + '\n')
        fileList.close()

        #subprocess.check_output('gdalbuildvrt -separate -input_file_list "{}" "{}"'.format(filename, outvrt), shell=True)
        vrtopts=gdal.BuildVRTOptions(separate=True)
        gdal.BuildVRT(outvrt, img_files, options=vrtopts)

        return outvrt, indates[idx_sort], img_files

    def get_size(self):
        with rasterio.open(self.file, 'r') as src:
            return src.height, src.width

    def get_crs(self):
        with rasterio.open(self.file, 'r') as src:
            return src.crs

    def get_transform(self):
        with rasterio.open(self.file, 'r') as src:
            return src.transform

    def get_pixel_position(self, coord_x, coord_y):
        with rasterio.open(self.file, 'r') as src:
            return src.index(coord_x, coord_y)


class Tile:
    '''
    An abstract parent class for each S2 tile containing references to where
    the tile imagery can be found
    Respective VRTs are created automatically each time a new datasource is
    added to this class
    '''

    def __init__(self, tile, vrt_dir, startdate, enddate, overwrite=False):

        self.id = tile
        self.datasources = {}
        self._startdate = pd.to_datetime(startdate)
        self._enddate = pd.to_datetime(enddate)
        self._overwrite = overwrite
        self._vrtdir = vrt_dir

        if not os.path.exists(vrt_dir):
            os.makedirs(self._vrtdir, exist_ok=True)

        self.datetimeindex = self._get_datetimeindex()

    def add_datasource(self, name, imgdir='', inpattern='', infiles=None,
                       outpattern='', dtype=np.float,
                       operation='1*x', nodata=None, dropna=False, mask=None):
        '''
        Method to add a new datasource to the tile object
        :param name: the name of the datasource (eg "s1_asc_vv")
        :param imgdir: the directory where the imagery is located
        :param inpattern: a datasource specific pattern based on which the
        list of files will be retrieved
        :param infiles: optional list of input files: cannot be specified
        together with imgdir and inpattern
        :param outpattern: basename of the output vrt to be created
        :param dtype: datatype of the imagery
        :param operation: optional numpy-like operation to be performed on the
        data when a window is requested
        :param nodata: input nodata value to be considered
        :param dropna: whether or not no data acquisitions should be dropped
        from the input stack
        :param mask: if not None, 'mask' refers to an existing datasource in
        the tile that serves as the mask for the new datasource
        :return:
        '''
        if infiles is not None:
            assert inpattern == '', ("A list of infiles cannot be provided"
                                     " together with an input pattern!")
        if infiles is not None:
            assert type(infiles) == list, "Provided infiles should be a list!"
            assert len(
                infiles) != 0, "Infiles should contain at least one image!"
        if mask is not None and mask not in self.datasources:
            raise ValueError(
                ('Provided mask "{}" does not exist in Tile object:'
                 ' first add the required mask!').format(mask))

        self.datasources[name] = Vrt(imgdir, self._vrtdir, self._startdate,
                                     self._enddate, inpattern, infiles,
                                     outpattern + '_' + self.id + '_',
                                     dtype=dtype, operation=operation,
                                     nodata=nodata, dropna=dropna, mask=mask)

        # Make sure the mask has the same acquisition dates
        if mask is not None:
            assert np.array_equal(
                self.datasources[name].dates, self.datasources[mask].dates)

    def get_datasource(self, name):
        if name not in self.datasources:
            log.error(
                'Datasource "{}" not found in this Tile object!'.format(name))
            raise Exception(
                'Datasource "{}" not found in this Tile object!'.format(name))
        return self.datasources[name]

    def list_datasources(self):
        return list(self.datasources.keys())

    def _get_datetimeindex(self):
        return pd.DatetimeIndex(pd.date_range(start=self._startdate,
                                              end=self._enddate, freq='5D'))


class Window:
    '''
    Abstract class that deals with individual windows (square blocks of data)
    that are input to the neural nets
    '''

    def __init__(self, window, tile, tlength='30D'):
        '''

        :param window:
        :param dim:
        :param tile:
        :param tlength: one-sided length of stack
        '''

        if not isinstance(tile, Tile):
            log.error('"tile" argument should be instance of Tile')
            raise Exception('"tile" argument should be instance of Tile')

        self.window = window
        self.dim = window[0][1] - window[0][0]
        self.tile = tile
        self.data = {}
        self.tlength = tlength
        self._hasdata = False
        self._clearacquisitions = None
        self._datastacks = None
        self._datastacks_array = None

    def _read_window_data(self):
        '''
        method that initiates the loading of all input data from the available
        datasources for this window
        the result is stored in the object as xarrays
        :return:
        '''

        log.info('Loading data for window {} ...'.format(self.window))

        datasources = self.tile.list_datasources()

        for datasource in datasources:

            datasourceObj = self.tile.get_datasource(datasource)
            operation = datasourceObj.operation

            log.info('Reading datasource: {}'.format(datasource))

            with rasterio.open(datasourceObj.file, 'r') as src:

                bands = src.count
                data = np.empty((self.dim, self.dim, bands)
                                ).astype(datasourceObj.dtype)
                local_namespace = {'x': data}

                for band in range(bands):
                    data[:, :, band] = src.read(band + 1, window=self.window)

                if datasourceObj.nodata is not None:
                    data[data == datasourceObj.nodata] = np.nan
                    idxnan = np.where(
                        np.sum(np.isnan(data), axis=(0, 1)) > 0)[0]
                    # If one pixel is nodata, then the whole window is
                    # unreliable so we set it to NaN
                    data[:, :, idxnan] = np.nan

                self.data[datasource] = xr.DataArray(
                    eval(operation, _GLOBAL_NAMESPACE, local_namespace),
                    coords=[np.arange(self.dim),
                            np.arange(self.dim), datasourceObj.dates],
                    dims=['x', 'y', 't'])

                if datasourceObj.dropna:
                    self.data[datasource] = self.data[datasource].dropna('t')

        # Now another HACK: sometimes, a pixel in S2 radiometry is NaN, while
        # the mask is 0 or 1. If at a later stage we forward impute NaN values,
        # then S2 radiometry and its mask become out of synch for those images!
        # So let's put all mask pixels to NaN where radiometry is NaN as well
        if 's2_b04' in self.data:
            self.data['s2_mask'].values[np.isnan(self.data['s2_b04'])] = np.nan
        if 's2_b08' in self.data:
            self.data['s2_mask'].values[np.isnan(self.data['s2_b08'])] = np.nan

        # Yet another HACK: if a pixel is NaN in one of the two S2 bands we
        # use to calculate NDVI, we should put it to NaN in the other one
        # as well
        if 's2_b04' in self.data:
            self.data['s2_b04'].values[np.isnan(self.data['s2_b08'])] = np.nan
        if 's2_b08' in self.data:
            self.data['s2_b08'].values[np.isnan(self.data['s2_b04'])] = np.nan

        self._hasdata = True

    def find_clear_acquisitions(self):
        '''
        method to find all sentinel-2 dates during which this
        particular window was clear (as defined by the s2_masks)
        :return:
        '''
        if not self._hasdata:
            self._read_window_data()
        if 's2_mask' not in self.tile.list_datasources():
            raise Exception(
                ('"s2_mask" should be in datasources'
                 ' to find clear acquisitions!'))
        # A clear acquisition is currently defined as having max 2% invalid
        # pixels, which we treat as false positives
        clear = (self.data['s2_mask'].sum(dim=['x', 'y']) <= int(
            0.02 * self.dim * self.dim)) & (
                np.sum(np.isnan(self.data['s2_mask'].values),
                       axis=(0, 1)) == 0)
        clear = clear[clear == 1]
        clearacquisitions = list(clear.indexes['t'])

        # Only look at clear acquisitions that are a certain time
        # after start date (tslength + 10 days) so we can
        # properly built a temporal stack
        self._clearacquisitions = [date for date in clearacquisitions if
                                   date >= self.tile._startdate +
                                   pd.to_timedelta(
                                       str(int(self.tlength[:-1]) + 10) + 'D')]

        log.info('Found {} clear acquisitions!'.format(
            len(self._clearacquisitions)))

        return list(clear.indexes['t'])

    def get_raw_arrays(self, datasource):
        """
        Method to return raw numpy arrays from a particular datasource
        :return:
        """
        if not self._hasdata:
            self._read_window_data()
        if datasource not in self.data:
            raise ValueError(
                "Datasource {} not found in window!".format(datasource))
        log.info('Getting data as raw arrarys ...')
        return np.array(self.data[datasource])

    def get_datastacks(self):
        '''
        method that retrieves all "matched" datastacks for the clear S2
        acquisition dates. The method does not just
        load data of individual datasources into the object, but takes
        care of temporal resampling all sources into
        one consistent datastack.
        :return:
        '''
        if self._datastacks is not None:
            return self._datastacks
        if not self._hasdata:
            self._read_window_data()
        if self._clearacquisitions is None:
            self.find_clear_acquisitions()
        if len(self._clearacquisitions) == 0:
            log.warning(
                'No clear acquisitions found for {}!'.format(self.window))
            return None

        datastacks = {}

        for acquisitiondate in self._clearacquisitions:
            log.info('Working on acquisition date: {}'.format(acquisitiondate))
            datastacks[acquisitiondate] = {}
            output_index = pd.date_range(acquisitiondate -
                                         pd.to_timedelta(self.tlength),
                                         acquisitiondate +
                                         pd.to_timedelta(self.tlength),
                                         freq='5D')

            for datasource in self.data.keys():
                log.info('Working on: {}'.format(datasource))

                if self.data[datasource].t.values[0] > output_index[0]:
                    # We need to implement an additional check here, as in a
                    # rare remaining case it is still possible
                    # that the start time of the source data is beyond the
                    # requested start time of the stack which
                    # would result in an error. Need to skip the entire window
                    # as an easy fix
                    log.warning(
                        ('Start date of {} is past the requested stack'
                         ' start time!').format(datasource))
                    datastacks[acquisitiondate][datasource] = None
                    continue

                # First ffill is to forward fill NaN values
                # Second ffill method is to forward fill the acquisition
                # indexes to the new 5-daily index
                # TODO: check what the behaviour is when multiple acquisitions
                # take place within the 5D window
                datastacks[acquisitiondate][datasource] = self.data[
                    datasource].ffill(dim='t').sel(
                        {'t': output_index}, method='ffill')

        self._datastacks = datastacks
        return self._datastacks

    def get_datastacks_as_array(self, nracquisitions=None):
        '''
        comparable method as self.get_datastacks, but the result
        is returned as numpy arrays instead of xarrays
        :param nracquisitions: optional maximum number of acquisitions to
        extract
        :return:
        '''

        if self._datastacks_array is not None:
            return self._datastacks_array
        if self._datastacks is None:
            self.get_datastacks()

        datastacks_arrays = {}

        if len(self._clearacquisitions) == 0:
            log.warning(
                'No clear acquisitions found for {}!'.format(self.window))
            return None

        if nracquisitions is not None:
            nracquisitions = int(nracquisitions)
            if len(self._clearacquisitions) >= nracquisitions:
                print(('Selecting {} acquisitions from the'
                       ' available {} ...').format(
                    nracquisitions, len(self._clearacquisitions)))
                clearacquisitions = np.random.choice(
                    self._clearacquisitions,
                    size=nracquisitions,
                    replace=False)
            else:
                clearacquisitions = self._clearacquisitions
        else:
            clearacquisitions = self._clearacquisitions

        for acquisitiondate in clearacquisitions:
            log.info('Working on acquisition date: {}'.format(acquisitiondate))
            datastacks_arrays[acquisitiondate] = {}

            # ANOTHER HACK: We don't want to return windows largely over water,
            # so if mean NDVI at acquisition date is below zero, we return
            # None and skip the window
            centerstep = int(
                self._datastacks[
                    acquisitiondate]['s2_b04'][:, :, ].shape[2]/2.)
            s2_b04 = self._datastacks[
                acquisitiondate]['s2_b04'][:, :, centerstep]
            s2_b08 = self._datastacks[
                acquisitiondate]['s2_b08'][:, :, centerstep]
            ndvi = np.divide((s2_b08-s2_b04), (s2_b08+s2_b04))
            if np.nanmean(ndvi) < 0:
                log.warning(
                    'Window {} has a mean NDVI below zero -> skipping'.format(
                        self.window))
                return None

            for datasource in self._datastacks[acquisitiondate].keys():
                log.info('Working on: {}'.format(datasource))

                if self._datastacks[acquisitiondate][datasource] is None:
                    # We've put it to None probably in an earlier step.
                    # Skipping this window as it is unreliable
                    log.warning(
                        ('Window {} has an issue (datastack is None for {})'
                         ' -> skipping').format(self.window, datasource))
                    return None

                if np.nansum(self._datastacks[acquisitiondate][
                        datasource].values) == 0:
                    # The datastack has all zeroes for this variable which
                    # means something has gone wrong so we skip this window
                    # entirely. This behaviour has been observed specifically
                    # at the border of a S2 tile where the last row of pixels
                    # were No Data consistently.
                    # As we put windows with this behaviour entirely to no
                    # data, we end up here with an all zeroes window.
                    # Safe to skip.
                    log.warning(
                        ('Window {} has an issue (all-zeroes stack for {})'
                         ' -> skipping').format(self.window, datasource))
                    return None

                datastacks_arrays[acquisitiondate][datasource] = np.array(
                    self._datastacks[acquisitiondate][datasource])

                # if datasource != 's2_mask':
                # Need to rescale
                # datastacks_arrays[acquisitiondate][datasource] =
                # scalers.minmaxscaler(
                #    datastacks_arrays[acquisitiondate][datasource],
                # datasource)
                #  NOTE: CURRENTLY WE DO NO SCALING IN THIS PROCEDURE

        self._datastacks_array = datastacks_arrays
        return self._datastacks_array

    def get_datastacks_as_array_one_date(self, acquisitiondate):
        '''
        return datastack for one date as numpy arrays
        :return:
        '''

        if not self._hasdata:
            self._read_window_data()

        output_index = pd.date_range(acquisitiondate -
                                     pd.to_timedelta(self.tlength),
                                     acquisitiondate +
                                     pd.to_timedelta(self.tlength),
                                     freq='5D')
        datastacks = {}
        for datasource in self.data.keys():
            log.info('Working on: {}'.format(datasource))

            # We allow to predict for any desired day, so the stack needs first
            # forward imputation to get rid of NaN values,
            # then resampling to daily resolution
            # again forward filling the acquisitions and
            # finally selecting the date of interest
            # TODO: check if this works as expected!
            datastacks[datasource] = self.data[
                datasource].ffill(dim='t').resample(t='1D').ffill().sel(
                {'t': output_index}, method='ffill')

        datastacks_arrays = {}

        log.info('Requested acquisition date: {}'.format(acquisitiondate))

        for datasource in datastacks.keys():
            log.info('Working on: {}'.format(datasource))

            datastacks_arrays[datasource] = np.array(datastacks[datasource])
            # if datasource != 's2_mask':
            #    # Need to rescale
            #    datastacks_arrays[datasource] = scalers.minmaxscaler(
            #        datastacks_arrays[datasource], datasource)
            # NOTE: CURRENTLY WE DO NO SCALING IN THIS PROCEDURE

        return datastacks_arrays


def GetExtent(gt, cols, rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext = []
    xarr = [0, cols]
    yarr = [0, rows]

    for px in xarr:
        for py in yarr:
            x = gt[0] + (px * gt[1]) + (py * gt[2])
            y = gt[3] + (px * gt[4]) + (py * gt[5])
            ext.append([x, y])
        yarr.reverse()
    return ext


def ReprojectCoords(coords, src_srs, tgt_srs):
    ''' Rep roject a list of x,y coordinates.

        @type geom:     C{tuple/list}
        @param geom:    List of [[x,y],...[x,y]] coordinates
        @type src_srs:  C{osr.SpatialReference}
        @param src_srs: OSR SpatialReference object
        @type tgt_srs:  C{osr.SpatialReference}
        @param tgt_srs: OSR SpatialReference object
        @rtype:         C{tuple/list}
        @return:        List of transformed [[x,y],...[x,y]] coordinates
    '''
    trans_coords = []
    transform = osr.CoordinateTransformation(src_srs, tgt_srs)
    for x, y in coords:
        x, y, z = transform.TransformPoint(x, y)
        trans_coords.append([x, y])
    return trans_coords


def get_GeoInfo(rasterfile):
    '''
    Function returns the geo information of a rasterfile

    :param rasterfile: input rasterfile
    :return: geo_info dictionary with geo information
    '''
    ds = gdal.Open(rasterfile)
    gt = ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    ext = GetExtent(gt, cols, rows)

    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(ds.GetProjection())
    tgt_srs = src_srs.CloneGeogCS()
    geo_ext = np.array(ReprojectCoords(ext, src_srs, tgt_srs))

    geo_info = dict()

    geo_info['extent'] = ext
    geo_info['x_min'] = min(np.array(ext)[:, 0])
    geo_info['x_max'] = max(np.array(ext)[:, 0])
    geo_info['y_min'] = min(np.array(ext)[:, 1])
    geo_info['y_max'] = max(np.array(ext)[:, 1])
    geo_info['Xdim'] = ds.RasterXSize
    geo_info['Ydim'] = ds.RasterYSize
    geo_info['srs'] = src_srs.ExportToProj4()

    geo_info['geo_ext'] = geo_ext

    return geo_info


def write2tif(ex_file, outfile, outdata, **kwargs):
    '''
    function to write a numpy array to a geotiff based on information from
    another geotiff file
    :param ex_file: the example file to take the geo information from
    :param outfile: the output geotif file to create
    :param outdata: the array to be written to the file
    :param kwargs: various possible additional arguments
    :return:
    '''
    # Check if we know the arguments
    for key, value in kwargs.items():
        if key not in ['mask_flag', 'datatype', 'nodata_value']:
            raise ValueError(("Unrecognized argument in "
                              "calling GeoTIFF.write(): ") + key)

    if len(outdata.shape) == 2:
        nr_bands = 1
    else:
        nr_bands = outdata.shape[2]

    # ex_file should be TIFF file that has the proper geo information for
    # copying to new file
    log.info('Writing to GeoTIFF...'),
    # register all of the GDAL drivers
    gdal.AllRegister()
    # open the input image
    inDs = gdal.Open(ex_file)
    # read in the input file info
    rows = inDs.RasterYSize
    cols = inDs.RasterXSize
    # create the output image
    driver = gdal.GetDriverByName('GTiff')
    if 'datatype' in kwargs:
        outDs = driver.Create(outfile, cols, rows, nr_bands,
                              kwargs['datatype'], ['COMPRESS=DEFLATE'])
    else:
        outDs = driver.Create(outfile, cols, rows,
                              nr_bands, 1, ['COMPRESS=DEFLATE'])
    if outDs is None:
        log.info('Could not create output tiff file')
        sys.exit(1)
    if len(outdata.shape) == 2:
        outBand = outDs.GetRasterBand(1)
        # write the data
        #        pdb.set_trace()
        outBand.WriteArray(outdata, 0, 0)
        # flush data to disk and set the NoData value
        if 'nodata_value' in kwargs:
            outBand.SetNoDataValue(kwargs['nodata_value'])
        outBand.FlushCache()
    else:
        nr_bands = outdata.shape[2]
        for i in range(nr_bands):
            log.info('Band ' + str(i + 1))
            outBand = outDs.GetRasterBand(i + 1)
            # write the data
            outBand.WriteArray(outdata[:, :, i], 0, 0)
            # flush data to disk and set the NoData value
            if 'nodata_value' in kwargs:
                outBand.SetNoDataValue(kwargs['nodata_value'])
            outBand.FlushCache()

    # georerence the image and set the projection
    outDs.SetGeoTransform(inDs.GetGeoTransform())
    outDs.SetProjection(inDs.GetProjection())
    outDs = None
    inDs = None
    log.info('-> Done!')

    if 'mask_flag' in kwargs:
        if kwargs['mask_flag'] == 1:
            log.info('Masking output image based on shapefile ...')

            ID = os.path.splitext(ex_file)[0].split('_')[-1]
            mask_file = os.path.join(os.path.dirname(os.path.splitext(ex_file)[
                0]), 'masks', 'GAES_mask_' + ID + '.shp')
            if not os.path.isfile(mask_file):
                sys.exit(
                    'Required mask file does not exist !! -> ' + mask_file)

            layer_name = os.path.splitext(os.path.basename(mask_file))[0]

            log.info('Creating tempfile ...')
            tempfile = os.path.join(os.path.dirname(outfile), 'temp.tif')
            # register all of the GDAL drivers
            gdal.AllRegister()
            # open the input image
            inDs = gdal.Open(outfile)

            # Create a zeros-array
            array = np.zeros((rows, cols))

            # create the output temp image
            driver = gdal.GetDriverByName('GTiff')
            tempDs = driver.Create(tempfile, cols, rows, 1, gdal.GDT_Int16)
            if tempDs is None:
                log.info('Could not create temp tiff file')
                sys.exit(1)
            outBand = tempDs.GetRasterBand(1)
            # write the data
            outBand.WriteArray(array, 0, 0)
            del array
            # georeference the image and set the projection
            tempDs.SetGeoTransform(inDs.GetGeoTransform())
            tempDs.SetProjection(inDs.GetProjection())
            # flush data to disk and set the NoData value
            outBand.FlushCache()
            tempDs = None

            log.info('Rasterizing vector mask ...')
            # Rasterize the mask vector
            subprocess.call(
                'gdal_rasterize -a dummy -l {} {} {}'.format(layer_name,
                                                             mask_file,
                                                             tempfile),
                shell=True)

            # Read in the rasterized results
            tempDs = gdal.Open(tempfile)
            maskdata = tempDs.ReadAsArray()

            unmasked_data = inDs.ReadAsArray()

            masked_data = np.array(unmasked_data, dtype=float)
            masked_data[np.where(maskdata == 0)] = -9999

            log.info('Create masked output file ...')
            # Write the final masked result to output file

            outfile_maked = os.path.join(os.path.dirname(outfile),
                                         os.path.splitext(
                os.path.basename(
                    outfile))[0] + '_masked.tif')

            outDs = driver.Create(outfile_maked, cols, rows, 1, gdal.GDT_Int16)
            if outDs is None:
                log.error('Could not create output tiff file')
                sys.exit(1)
            outBand = outDs.GetRasterBand(1)
            # write the data
            outBand.WriteArray(masked_data, 0, 0)
            outBand.SetNoDataValue(-9999)
            del masked_data
            # georeference the image and set the projection
            outDs.SetGeoTransform(inDs.GetGeoTransform())
            outDs.SetProjection(inDs.GetProjection())
            # flush data to disk and set the NoData value
            outBand.FlushCache()
            outDs = None
            inDs = None

            # Delete tempfile
            tempDs = None
            os.remove(tempfile)


def temporalVRT(outvrt, indir, inpattern, startdate=None, enddate=None):

    import pandas as pd

    if os.path.exists(outvrt):
        os.remove(outvrt)
    log.info('Creating: {}'.format(outvrt))

    img_files = glob.glob(os.path.join(indir, inpattern))

    if len(img_files) == 0:
        log.warning('No img files found under: {}'.format(
            os.path.join(indir, inpattern)))
        return

    if startdate is not None and enddate is not None:

        indates = pd.to_datetime(np.array(
            [pd.to_datetime(
                os.path.basename(x).split('_')[0]) for x in img_files]))
        mask = (indates >= startdate) & (indates <= enddate)

        img_files = np.array(img_files)[mask]
        indates = indates[mask]
        idx_sort = np.argsort(indates)
        img_files = list(img_files[idx_sort])

    # Create a file that will hold the individual files
    filename = os.path.join(indir, 'temp_filelist.txt')

    if os.path.exists(filename):
        os.remove(filename)
    fileList = open(filename, 'w')

    for file in img_files:
        fileList.write(file + '\n')
    fileList.close()

    #subprocess.check_output('gdalbuildvrt -separate -input_file_list "{}" "{}"'.format(filename, outvrt), shell=True)
    vrtopts=gdal.BuildVRTOptions(separate=True)
    gdal.BuildVRT(outvrt, img_files, options=vrtopts)

    return


def rst_proj_check(rst_file_list, epsg_code):
    '''
    Checks whether all rasters in the list have the same CRS, equal to the one
    that is defined by input parameter
    epsg_gode.

    :param rst_file_list: list of raster file names to be checked
    :param epsg_code: epsg code of desired CRS
    :return: list of 0 or 1 stating whether CRS matches (1) or not (0)
    '''

    CRS_OK = []

    for rast_file in rst_file_list:
        # get extent and resolution
        # info = get_GeoInfo(rast_file)
        # get projection info
        d = gdal.Open(rast_file)
        d_proj = osr.SpatialReference(wkt=d.GetProjection())
        epsg_proj = d_proj.GetAttrValue("AUTHORITY", 1)
        if not epsg_proj == str(epsg_code):
            CRS_OK.append(0)
        else:
            CRS_OK.append(1)

    return CRS_OK


def rst_proj_check_warp(rst_file_list, epsg_code):
    '''
    Checks whether all rasters in the list have the same CRS, equal to the one
    that is defined by input parameter
    epsg_gode. If CRS doesn't match, rasters are warped.

    :param rst_file_list: list of raster file names to be checked
    :param epsg_code: epsg code of desired CRS
    :return: list of raster files all having the same CRS
    '''

    rast_file_list_adapted = []

    for rast_file in rst_file_list:
        # get extent and resolution
        # info = get_GeoInfo(rast_file)
        # get projection info
        d = gdal.Open(rast_file)
        d_proj = osr.SpatialReference(wkt=d.GetProjection())
        epsg_proj = d_proj.GetAttrValue("AUTHORITY", 1)
        if not epsg_proj == str(epsg_code):
            # do the resampling using gdal Warp
            outfile = rast_file[:-4] + '_' + str(epsg_code) + '.tif'
            if not os.path.exists(outfile):
                _ = gdal.Warp(outfile, rast_file,
                              options='-t_srs EPSG:{}'.format(str(epsg_code)))
            d = None
            rast_file_list_adapted.append(outfile)
        else:
            rast_file_list_adapted.append(rast_file)

    return rast_file_list_adapted


def clip2polyExt(source_file, poly, outfile):
    '''
    function to clip a .tiff raster using the extent of a shapefile containing
    a single polygon
    :param source_file: the raster file to be clipped
    :param poly: the shapefile containing one polygon defining the extent
    :param outfile: the output geotif file to create
    :return:
    '''

    # get extent from shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(poly, 0)  # 0 means read-only. 1 means writeable.
    layer = dataSource.GetLayer()
    ext = layer.GetExtent()

    ds = gdal.Open(source_file)
    win = [ext[0], ext[3], ext[1], ext[2]]
    _ = gdal.Translate(outfile, ds, projWin=win)
    ds = None


def poly2rast(polyfile, attribute_fields, snapraster, outfile):
    '''
    Convert polygon to raster (.tif) with pixel resolution same as snapraster

    :param polyfile: Shapefile to be converted
    :param attribute_fields: name of (numeric) attribute fields to be used for
    conversion
    :param snapraster: Raster (.tif) defining output pixel resolution and geo
    information
    :param outfile: Filename of the raster file to be created (including .tif
    extension)
    :return: /
    '''

    ###########################################################################
    # USING GEOPANDAS
    ###########################################################################

    # open the shapefile
    poly = gpd.read_file(polyfile)
    # get metadata from snapraster
    snaprst = rasterio.open(snapraster)
    meta = snaprst.meta.copy()
    meta.update(compress='lzw')
    meta.update(count=len(attribute_fields))

    # burn features in raster
    tempfile = outfile[:-4] + '_tmp.tif'
    with rasterio.open(tempfile, 'w+', **meta) as out:
        for i in range(len(attribute_fields)):
            out_arr = out.read(i+1)
            attribute = poly.filter([attribute_fields[i]], axis=1).squeeze()
            shapes = ((geom, value)
                      for geom, value in zip(poly.geometry, attribute))
            burned = features.rasterize(
                shapes=shapes, fill=-9999, out=out_arr,
                transform=out.transform)
            out.write_band(i+1, burned)

    # crop to extent of shapefile
    clip2polyExt(tempfile, polyfile, outfile)
    # delete full raster
    os.remove(tempfile)

def poly2rast_rastExtent(polyfile, attribute_fields, snapraster, outfile):
    '''
    Convert polygon to raster (.tif) with pixel resolution same as snapraster

    :param polyfile: Shapefile to be converted
    :param attribute_fields: name of (numeric) attribute fields to be used for conversion
    :param snapraster: Raster (.tif) defining output pixel resolution and geo information
    :param outfile: Filename of the raster file to be created (including .tif extension)
    :return: /
    '''

    ############################################################################################################
    # USING GEOPANDAS
    ############################################################################################################

    # open the shapefile
    poly = gpd.read_file(polyfile)
    # get metadata from snapraster
    snaprst = rasterio.open(snapraster)
    meta = snaprst.meta.copy()
    meta.update(compress='lzw')
    meta.update(count=len(attribute_fields))

    # burn features in raster
    with rasterio.open(outfile, 'w+', **meta) as out:
        for i in range(len(attribute_fields)):
            out_arr = out.read(i+1)
            attribute = poly.filter([attribute_fields[i]], axis=1).squeeze()
            if not isinstance(attribute, pd.Series):
                attribute = [attribute]
            shapes = ((geom, value) for geom, value in zip(poly.geometry, attribute))
            burned = features.rasterize(shapes=shapes, fill=-9999, out=out_arr, transform=out.transform)
            out.write_band(i+1, burned)


def clip_poly_by_raster(poly, raster, outfile):
    '''
    Clip polygon layer using raster extent (GDAL)
    Note: if both input layers don't share the same CRS, the shapefile will
    be reprojected

    :param poly: shapefile to be clipped
    :param raster: raster from which the extent is used for clipping
    :param outfile: full path to output file
    :return:
    '''

    src = gdal.Open(raster)
    proj = osr.SpatialReference(wkt=src.GetProjection())
    epsg = proj.GetAttrValue('AUTHORITY', 1)
    ulx, xres, xskew, uly, yskew, yres = src.GetGeoTransform()
    sizeX = src.RasterXSize * xres
    sizeY = src.RasterYSize * yres
    lrx = ulx + sizeX
    lry = uly + sizeY
    src = None

    # format the extent coords
    extent = '{0} {1} {2} {3}'.format(ulx, lry, lrx, uly)

    # reproject shapefile if necessary
    shape = gpd.read_file(poly)
    epsg_poly = shape.crs['init'][5:]
    if epsg_poly == epsg:
        poly_reproj = poly
    else:
        poly2 = shape.to_crs({'init': 'epsg:{}'.format(epsg)})
        poly_reproj = poly[:-4] + '_epsg{}.shp'.format(epsg)
        poly2.to_file(poly_reproj)

    # make clip command with ogr2ogr - default to shapefile format
    cmd = f'ogr2ogr {outfile} {poly_reproj} -overwrite -clipsrc {extent}'

    # call the command
    subprocess.call(cmd, shell=True)
