"""
Methods for creating new features.
"""

import re
import operator

from kcmn.series import uniqValReplace

def family_size(df):
    df['FamilySize'] = df['SibSp'] + df['Parch']


def name_length(df):
    df['NameLength'] = df['Name'].apply(lambda x: len(x))


def title(df):
    df['Title'], map = uniqValReplace(df['Name'].apply(__get_title))
    return map

family_id_mapping = {}
def family_id(df):
    family_id_mapping = {}
    family_ids = df.apply(__get_family_id, axis=1)
    family_ids[df['FamilySize'] < 3] = -1
    df['FamilyId'] = family_ids


# private

def __get_family_id(row):
    last_name = row['Name'].split(',')[0]
    f_id = '{0}{1}'.format(last_name, row['FamilySize'])

    if f_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)

        family_id_mapping[f_id] = current_id

    return family_id_mapping[f_id]


def __get_title(name):
    ts = re.search(' ([A-Za-z]+)\.', name)

    if ts:
        return ts.group(1)

    return ''