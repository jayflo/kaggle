
# Provides a map for replacing series values with integers.

# param srs: a pandas.Series

# returns a dict whose keys are the unique values of srs and whose values are
# consecutive integers from 0 to #{unique series values}


def uniqValMap(srs):
    return dict(list(map(
        lambda tup: tup[::-1],
        enumerate(srs.unique().tolist())
    )))


# Replaces the values of a series with integers.

# param srs: a pandas.Series

# returns a tuple whose first coordinate is a map from unique values of the series
# to integers, and whose second coordinate is srs with values replaced by the
# integers given in the map (integers range from 0 to #{unique series values})


def uniqValReplace(srs):
    map = uniqValMap(srs)

    for k, v in map.items():
        srs[srs == k] = v

    return srs, map
