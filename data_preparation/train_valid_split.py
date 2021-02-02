

#
# Train valid split
# Original idea https://www.kaggle.com/its7171/cv-strategy
# Thanks to @tito on kaggle
#

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import random

data = pd.read_pickle("whole-train.pickle")


max_timestamp_u = data[['user_id','timestamp']].groupby(['user_id']).agg(['max']).reset_index()
max_timestamp_u.columns = ['user_id', 'max_time_stamp']
MAX_TIME_STAMP = max_timestamp_u.max_time_stamp.max()

def rand_time(max_time_stamp):
    interval = MAX_TIME_STAMP - max_time_stamp
    rand_time_stamp = random.randint(0,interval)
    return rand_time_stamp
random.seed(42)
max_timestamp_u['rand_time_stamp'] = max_timestamp_u.max_time_stamp.apply(rand_time)
data = data.merge(max_timestamp_u, on='user_id', how='left')
data['virtual_time_stamp'] = data.timestamp + data['rand_time_stamp']

data.sort_values(['virtual_time_stamp', 'row_id'], inplace=True)
data.reset_index(inplace=True, drop=True)
data = data.drop(columns=['rand_time_stamp','virtual_time_stamp', 'max_time_stamp'])

#
# 95% for train and 5% for valid
#
N = len(data)
nvalid = int(0.05*N)

valid = data[-nvalid:]
train = data[:-nvalid]
del data

train.sort_values(['user_id', 'timestamp', 'encoded_content_id'], inplace=True)
train.reset_index(inplace=True, drop=True)
valid.sort_values(['user_id', 'timestamp', 'encoded_content_id'], inplace=True)
valid.reset_index(inplace=True, drop=True)

valid.to_pickle('valid.pickle')
train.to_pickle('train.pickle')