import numpy as np
from sklearn.preprocessing import MinMaxScaler

def s1():
    '''
    This function constructs manual minmax scalers for S1 variables
    '''

    scalers = dict()

    ranges = dict()
    ranges['VV'] = [-20, 0]
    ranges['VH'] = [-33, -8]
    ranges['angle'] = [29.98, 46]
    ranges['RVI'] = [0.2, 1]

    for band in ranges.keys():
        scalers[band] = MinMaxScaler(feature_range=(-1, 1)).fit(np.array(ranges[band]).astype(float).reshape(-1, 1))

    return scalers

def s2():
    '''
    This function constructs manual minmax scalers for S2 variables
    '''

    scalers = dict()

    ranges = dict()
    ranges['B1'] = [0, 0.22]
    ranges['B2'] = [0, 0.17]
    ranges['B3'] = [0, 0.20]
    ranges['B4'] = [0, 0.22]
    ranges['B5'] = [0, 0.27]
    ranges['B6'] = [0.05, 0.6]
    ranges['B7'] = [0.05, 0.64]
    ranges['B8'] = [0.05, 0.68]
    ranges['B8A'] = [0.05, 0.68]
    ranges['B9'] = [0.05, 0.65]
    ranges['B11'] = [0, 0.41]
    ranges['B12'] = [0, 0.37]
    ranges['ndvi'] = [-0.2, 1]

    for band in ranges.keys():
        scalers[band] = MinMaxScaler(feature_range=(-1, 1)).fit(np.array(ranges[band]).astype(float).reshape(-1, 1))

    return scalers


def minmaxscaler(data, source,
                 minscaled=-1, maxscaled=1):
    ranges = {}
    ranges['s2_fapar'] = [0.07, 1]
    ranges['s2_ndvi'] = [-0.08, 1]
    ranges['s1_asc_vv'] = [-20, -2]
    ranges['s1_asc_vh'] = [-33, -8]
    ranges['s1_des_vv'] = [-20, -2]
    ranges['s1_des_vh'] = [-33, -8]
    ranges['s2_mask'] = [0, 1]

    if source not in ranges.keys():
        raise Exception('Datasource "{}" not in known scalers!'.format(source))

    # Scale between minscaled and maxscaled
    datarescaled = (
        (maxscaled - minscaled) *
        (data - ranges[source][0]) /
        (ranges[source][1] - ranges[source][0])
        + minscaled
    )

    return datarescaled


def minmaxunscaler(data, source,
                   minscaled=-1, maxscaled=1):
    ranges = {}
    ranges['s2_fapar'] = [0.07, 1]
    ranges['s2_ndvi'] = [-0.08, 1]
    ranges['s1_asc_vv'] = [-20, -2]
    ranges['s1_asc_vh'] = [-33, -8]
    ranges['s1_des_vv'] = [-20, -2]
    ranges['s1_des_vh'] = [-33, -8]
    ranges['s2_mask'] = [0, 1]

    if source not in ranges.keys():
        raise Exception('Datasource "{}" not in known scalers!'.format(source))

    # Unscale
    dataunscaled = (
        (data - minscaled) *
        (ranges[source][1] - ranges[source][0]) /
        (maxscaled - minscaled) +
        ranges[source][0]
    )

    return dataunscaled


if __name__ == '__main__':
    pass
