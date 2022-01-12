#!/usr/bin/env python3

'''
Original script by: Kris Vanhoof and Dominique Haesen
'''

import datetime
import os
import sys
import timeit, time
import traceback
import requests

import numpy as np
import pandas as pd
import geopandas as gpd

import shapely.geometry
import dateutil.parser

import dataclient


def _make_multipolygon(obj):

    # If obj is a Polygon, convert it to a MultiPolygon

    if obj.type == 'Polygon':
        poly = shapely.geometry.shape(obj)
        return shapely.geometry.MultiPolygon([poly])
    else:
        return obj


def retrieve_timeseries(layer, df, start_date=None, end_date=None,
                        buffer_distance=0.0, buffer_crs='epsg:32631',
                        ts_endpoint='https://proba-v-mep.esa.int/api/timeseries/v1.0/ts/',
                        check_overlap=False):

    # Default start/end range is 2017-01-01 to now

    if start_date is None:
        start_date = datetime.datetime(2017, 1, 1, tzinfo=datetime.timezone.utc)
    if end_date is None:
        end_date = datetime.datetime.utcnow()

    # Apply a buffer to the parcel shapes

    df = df.copy()

    #if buffer_distance != 0.0:
    # WE'LL DO BUFFERING ANYWAY, AS IT FIXES SOME OF THE GEOMETRY ERRORS

    print('Buffering polygons ...')

    # Convert to CRS for buffering if needed

    df_crs = df.crs['init']
    if df_crs != buffer_crs:
        df = df.to_crs({'init': buffer_crs})

    # Do the actual buffering

    # resolution: amount of segments used to approximate circle
    # JOIN_STYLE 2: miter
    df.geometry = df.geometry.buffer(buffer_distance, resolution=2, join_style=2)

    # Remove empty parcel shapes

    df = df[~df.geometry.is_empty]

    # Convert to original CRS if needed

    if df_crs != buffer_crs:
        df = df.to_crs({'init': df_crs})

    # Make sure all geometries have the same type

    df.geometry = df.geometry.apply(_make_multipolygon)

    # Check if no invalid geometries are still in the dataframe
    df = df.loc[df.geometry.is_valid.index]

    # check if there are overlapping polygons in the dataframe
    #
    if check_overlap:
        print('Checking for overlapping polygons ...')
        geoms = df['geometry'].tolist()
        overlapping_geoms = []
        for x in range(0, len(geoms) - 1):
            for y in range(x + 1, len(geoms)):
                if geoms[x].intersection(geoms[y]).area > 0:
                    overlapping_geoms.append((x, y))
    # #
    # # create lists of non overlapping polygons
    # #
    # field_lists_without_overlap = []
    # zipped_lst_fieldIds = list(zip(df.index.tolist(), range(0, len(df.index.tolist()))))
    # remaining_fieldIds = zipped_lst_fieldIds
    # while len(remaining_fieldIds) > 0:
    #     remaining_fieldIds_indices = list(map(lambda f: f[1], remaining_fieldIds))
    #     overlapping_field_indices = set()
    #     fields_without_overlap = remaining_fieldIds
    #     for x in overlapping_geoms:
    #         if (x[0] in remaining_fieldIds_indices and
    #                 x[1] in remaining_fieldIds_indices and
    #                 x[0] not in overlapping_field_indices):
    #             fields_without_overlap = [i for i in fields_without_overlap if i[1] != x[1]]
    #             overlapping_field_indices.add(x[1])
    #     field_lists_without_overlap.append(fields_without_overlap)
    #     remaining_fieldIds = [i for i in remaining_fieldIds if i[1] in overlapping_field_indices]

    #
    # Subset the dataframe only on the non-overlapping fields
    #

        if len(overlapping_geoms) > 0: sys.exit('There are overlapping polygons in the dataframe: we cannot currently handle this properly!')


    # Query the timeseries webservice
    ts_df = dataclient.get_timeseries_n_features(
        features_df=df,
        layername=layer if type(layer) == str else layer.tsservice_layer_id,
        startdate=start_date,
        enddate=end_date,
        endpoint=ts_endpoint)

    avg = pd.DataFrame.from_dict(ts_df)

    avg[pd.isnull(avg)] = np.nan

    return avg


def split(df, chunk_size):

    # Calculate the number of chunks

    num_chunks = (df.shape[0]-1) // chunk_size + 1

    # Calculate the split positions

    indices = range(chunk_size, num_chunks*chunk_size, chunk_size)

    return np.split(df, indices)


def write_part(df, name, dir, format='csv'):

    # Columns names are datetime[ns] but should be strings

    df.columns = df.columns.astype(str)

    # Write to file in the specified file format

    if format == 'parquet':
        df = df.astype(float)
        df.to_parquet(os.path.join(dir, name))
    else:
        df.to_csv(os.path.join(dir, name))

def append_part(df, name, dir, format='csv'):
    if format != 'csv': raise NotImplementedError('appending only implemented for CSV files!')

    existing_data = pd.read_csv(os.path.join(dir, name), index_col=0, parse_dates=True)
    if len(set(existing_data.columns.tolist()).intersection(set(df.columns.tolist()))) != len(df.columns.tolist()):
        sys.exit('Column names do not match -> cannot append data!')
    merged_data = pd.concat([existing_data, df])

    # Columns names are datetime[ns] but should be strings
    merged_data.columns = merged_data.columns.astype(str)

    # Write to file in the specified file format
    os.remove(os.path.join(dir, name))
    merged_data.to_csv(os.path.join(dir, name))


def check_new_data(layer, chunk_file, start_date, end_date):
    if layer.startswith('S1'): layer_str = 'CGS_S1_GRD_SIGMA0_L1'
    elif layer.startswith('S2'): layer_str = 'CGS_S2_FAPAR'

    all_dates = sorted(set(
        [pd.to_datetime(x).date() for x in requests.get("https://proba-v-mep.esa.int/api/catalog/v2/" + layer_str + "/times").json() if
         pd.to_datetime(x).date() >= start_date.date() and pd.to_datetime(x).date() <= end_date.date()]))

    chunk_dates = pd.read_csv(chunk_file, index_col=0, parse_dates=True).index.tolist()
    chunk_dates = [x.date() for x in chunk_dates]

    if all_dates[-1] > chunk_dates[-1]:
        new_start_date = chunk_dates[-1]
        new_end_date = all_dates[-1]
        return [new_start_date, new_end_date]
    else: return None


def main(shp_file, output, pattern, start_date, end_date,
         ts_endpoint='http://tsviewer-rest-test.vgt.vito.be:8080/v1.0/ts/',
         max_retries=10, wait_time=60, layer='S2_FAPAR', ts_user='user', ts_password='invalid',
         buffer_distance=0.0, buffer_crs='epsg:32631', output_format='csv', chunk_size=5000, shp_index='CODE_OBJ',
         overwrite=False, do_append=False, selection_attribute=None, selection_values=None):

    os.makedirs(output, exist_ok=True)

    # Get the layer name
    layer = layer

    # Get the shapefile
    shp_file = shp_file

    print('--Processing .shp file {}'.format(shp_file))

    t0 = timeit.default_timer()

    df = gpd.read_file(shp_file)

    # Subset if needed
    if selection_attribute is not None and selection_values is not None:
        if selection_attribute not in df: sys.exit('Selection attribute not in dataframe!')
        if type(selection_values) is not list: sys.exit('Selection values should be a list!')
        print('Subsetting on attribute {} for values: {}'.format(selection_attribute, selection_values))
        df = df[df[selection_attribute].isin(selection_values)]

    df.index = df[shp_index]

    t1 = timeit.default_timer()

    print('--Loaded {} parcels from {}'.format(df.shape[0], shp_file))

    chunk_dfs = split(df, chunk_size)

    for j in range(len(chunk_dfs)):

        print('----Processing chunk [{}/{}]'.format(j+1, len(chunk_dfs)))

        if output_format == 'parquet':
            out_name = '{}-{}-{}-{}-{:04d}'.format(layer, start_date.replace('-',''),
                                                         end_date.replace('-',''), pattern, j) + '.parquet'
        else:
            out_name = '{}-{}-{}-{}-{:04d}'.format(layer, start_date.replace('-', ''),
                                                   end_date.replace('-', ''), pattern, j) + '.csv'

        append = False
        if os.path.exists(os.path.join(output, out_name)) and not overwrite:
            if not do_append:
                print('Part exists -> skipping ({})'.format(out_name))
                continue
            # Check if we have new data to append to the existing file
            new_dates = check_new_data(layer, os.path.join(output, out_name),
                            dateutil.parser.parse(start_date),
                            dateutil.parser.parse(end_date))
            if new_dates is not None:
                print('Part exists but new data found -> need to append ... ({})'.format(out_name))
                append = True
            else:
                print('Part exists -> skipping ({})'.format(out_name))
                continue

        t3 = timeit.default_timer()

        attempt = 0
        while True:
            attempt += 1
            try:
                if not append:
                    df_avg = retrieve_timeseries(layer, chunk_dfs[j],
                                                    dateutil.parser.parse(start_date),
                                                    dateutil.parser.parse(end_date),
                                                    buffer_distance,
                                                    buffer_crs,
                                                    ts_endpoint)
                else:
                    df_avg = retrieve_timeseries(layer, chunk_dfs[j],
                                                 pd.to_datetime(new_dates[0]),
                                                 pd.to_datetime(new_dates[1]),
                                                 buffer_distance,
                                                 buffer_crs,
                                                 ts_endpoint)

            except Exception:
                df_avg = None
                print(traceback.format_exc())

            if df_avg is not None:

                t4 = timeit.default_timer()

                if not append: write_part(df_avg, out_name, output, output_format)
                else: append_part(df_avg, out_name, output, output_format)

                t5 = timeit.default_timer()

                print('----Processing chunk took: {:0.2f}s ({:0.2f}s + {:0.2f}s)'.format(t5-t3, t4-t3, t5-t4))

                break

            if max_retries <= attempt:
                print('Retrieval ultimately failed!')
                break

            print('Retrieval failed current attempt -> retrying in {} seconds'.format(wait_time*attempt))
            time.sleep(wait_time * attempt)

    t2 = timeit.default_timer()

    print('--Processing .shp file took: {:0.2f}s ({:0.2f}s + {:0.2f}s)'.format(t2-t0, t1-t0, t2-t1))

if __name__ == '__main__':
    main()

