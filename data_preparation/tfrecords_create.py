

#
# Creating dataset in tfrecord format for training on TPU
#
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import random
import tensorflow as tf
import os 

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_seq_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_seq_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def build_data_example(row, columns, dtypes):
    feature = {
        colname: _int64_seq_feature([row[colname]]) if dtype == 'scalar_int'
                else _int64_seq_feature(row[colname]) if dtype=='int' 
                else _float_seq_feature(row[colname]) 
        for colname, dtype in zip(columns, dtypes)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

#
# Creating tfrecords for train
#
train = pd.read_pickle("train.pickle")
train = train.fillna(0)

### We must cut train in 2 to avoid out of memory when aggregating ...
mid_user = train.user_id.median()
train_agg = train[train.user_id < mid_user].groupby('user_id')[['timestamp', 'time_lag', 'encoded_content_id',  'user_answer',
                                      'answered_correctly', 'question_elapsed_time', 'question_had_explanation']].agg(list).reset_index()
train_agg.to_pickle('train_agg_1.pickle')
del train_agg
train_agg = train[train.user_id >= mid_user].groupby('user_id')[['timestamp', 'time_lag', 'encoded_content_id',  'user_answer',
                                      'answered_correctly', 'question_elapsed_time', 'question_had_explanation']].agg(list).reset_index()

train_agg = train_agg.append(pd.read_pickle('train_agg_1.pickle'))
os.remove("train_agg_1.pickle")

N_TRAIN = len(train_agg)
N_SHARD = 16
N_PER_SHARD = 1+ (N_TRAIN // N_SHARD)

os.makedirs('tfrecords/train', exist_ok=True)

for i in range(N_SHARD):
    print(f'Writing shard {i}', flush=True)
    with tf.io.TFRecordWriter(f"tfrecords/train/train_{i}.tfrecords") as writer:
        for _,row in train_agg[(i*N_PER_SHARD):((i+1)*N_PER_SHARD)].iterrows():
            tf_example = build_data_example(row, 
                                        columns= ['user_id', 'timestamp', 'time_lag', 'encoded_content_id',  'user_answer',
                                                 'answered_correctly', 'question_elapsed_time', 'question_had_explanation'],
                                        dtypes = ['scalar_int', 'int', 'float', 'int', 'int', 'int', 'float', 'int']
                                      )
            writer.write(tf_example.SerializeToString())

del train_agg
#
# Creating tfrecords for valid
#
valid = pd.read_pickle("valid.pickle")
valid = valid.fillna(0)

#
# For each user in the valid set, retrieve his/her previous activities in the train set
#  in_valid_set is a boolean flag to determine if the activity in the valid set
#
valid_user = valid[['user_id']].drop_duplicates()
train['in_valid_set'] = False
valid['in_valid_set'] = True
valid_in_train = valid_user.merge(train, on='user_id', how='inner')
valid = valid.append(valid_in_train)
valid = valid.sort_values(['user_id', 'timestamp', 'encoded_content_id']).reset_index(drop=True)

valid_agg = valid.groupby('user_id')[['timestamp', 'time_lag', 'encoded_content_id', 'user_answer',
                                      'answered_correctly', 'question_elapsed_time', 
                                      'question_had_explanation', 'in_valid_set']].agg(list).reset_index()

N_VALID = len(valid_agg)
N_SHARD = 4
N_PER_SHARD = 1+ (N_VALID // N_SHARD)
os.makedirs('tfrecords/valid', exist_ok=True)

for i in range(N_SHARD):
    print(f'Writing shard {i}', flush=True)
    with tf.io.TFRecordWriter(f"tfrecords/valid/valid_{i}.tfrecords") as writer:
        for _,row in valid_agg[(i*N_PER_SHARD):((i+1)*N_PER_SHARD)].iterrows():
            tf_example = build_data_example(row, 
                                        columns=['user_id', 'timestamp', 'time_lag', 'encoded_content_id',  'user_answer',
                                                 'answered_correctly', 'question_elapsed_time', 
                                                 'question_had_explanation', 'in_valid_set'],
                                        dtypes = ['scalar_int', 'int', 'time_lag', 'int', 'int', 'int', 'float', 'int', 'int']
                                      )
            writer.write(tf_example.SerializeToString())
del train
del valid
del valid_agg
#
# Build tfrecords dataset for the whole train data
#
whole_train = pd.read_pickle("whole-train.pickle")
whole_train = whole_train.fillna(0)

mid_user = whole_train.user_id.median()
whole_train_agg = whole_train[whole_train.user_id < mid_user].groupby('user_id')[['timestamp', 'time_lag', 'encoded_content_id',  'user_answer',
                                      'answered_correctly', 'question_elapsed_time', 'question_had_explanation']].agg(list).reset_index()
whole_train_agg.to_pickle('whole_train_agg_1.pickle')
del whole_train_agg
whole_train_agg = whole_train[whole_train.user_id >= mid_user].groupby('user_id')[['timestamp', 'time_lag', 'encoded_content_id',  'user_answer',
                                      'answered_correctly', 'question_elapsed_time', 'question_had_explanation']].agg(list).reset_index()
del whole_train

whole_train_agg = whole_train_agg.append(pd.read_pickle('whole_train_agg_1.pickle'))

os.remove('whole_train_agg_1.pickle')

N_TRAIN = len(whole_train_agg)
N_SHARD = 16
N_PER_SHARD = 1+ (N_TRAIN // N_SHARD)
os.makedirs('tfrecords/whole_train', exist_ok=True)

for i in range(N_SHARD):
    print(f'Writing shard {i}', flush=True)
    with tf.io.TFRecordWriter(f"tfrecords/whole_train/whole_train_{i}.tfrecords") as writer:
        for _,row in whole_train_agg[(i*N_PER_SHARD):((i+1)*N_PER_SHARD)].iterrows():
            tf_example = build_data_example(row, 
                                        columns=['user_id', 'timestamp', 'time_lag', 'encoded_content_id',  'user_answer',
                                                 'answered_correctly', 'question_elapsed_time', 'question_had_explanation'],
                                        dtypes = ['scalar_int', 'int', 'float', 'int', 'int', 'int', 'float', 'int']
                                      )
            writer.write(tf_example.SerializeToString())