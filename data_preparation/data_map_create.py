
#
# Create numpy data_map to submission models on kaggle
#

import tensorflow as tf
import numpy as np
import pickle 

def build_parse_data_function(sequence_length, columns=['timestamp'], dtypes=['int']):
    feature_description = {
        colname: (tf.io.FixedLenFeature([], tf.int64, default_value=0) if dtype == 'scalar_int'
                  else tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True) 
                  if dtype=='int' else tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)) 
        for colname, dtype in zip(columns, dtypes)
    }
    
    def _parse_fn(example_proto):
        parsed_data = tf.io.parse_single_example(example_proto, feature_description)
        length = tf.shape(parsed_data['timestamp'])[0]
        diff_length = tf.constant(sequence_length) -  length
        parsed_data['non_padding_mask'] = tf.ones_like(parsed_data['timestamp'], 
                                                       dtype = tf.float32)
        result = {k:parsed_data[k] for k in {
                    'timestamp',
                    'time_lag',
                    'encoded_content_id',
                    'user_answer',
                    'answered_correctly',
                    'question_elapsed_time',
                    'question_had_explanation',
                    'non_padding_mask'
                }}
        if tf.greater_equal(diff_length, 0):
            result = {
                k:tf.pad(v, [[diff_length, 0]]) for k, v in result.items()
            }
        else:
            result = {
                k:tf.slice(v, [-diff_length], [sequence_length]) for k,v in result.items()
            }
        
        result = {
            k:tf.ensure_shape(v, shape=(sequence_length,)) for k,v in result.items()
        }
        result['user_id'] = parsed_data['user_id']
        return result
    
    return _parse_fn


SEQUENCE_LENGTH = 512
NUM_USERS = 450000
shape = (NUM_USERS, SEQUENCE_LENGTH)
data_map = {
    'data' : {
        'timestamp': np.zeros(shape = shape, dtype = np.int64),
        'time_lag':np.zeros(shape = shape, dtype = np.float32),
        'encoded_content_id':np.zeros(shape = shape, dtype = np.int32),  
        'user_answer':np.zeros(shape = shape, dtype = np.int8),
        'answered_correctly':np.zeros(shape = shape, dtype = np.int8), 
        'question_elapsed_time':np.zeros(shape = shape, dtype = np.float32), 
        'question_had_explanation':np.zeros(shape = shape, dtype = np.int8),
        'non_padding_mask': np.zeros(shape = shape, dtype = np.int8)
        },
    'index':{
        
    },
    'next_index': 0
}

parse_to_seq = build_parse_data_function(SEQUENCE_LENGTH,
                          columns = ['user_id', 'timestamp', 'time_lag', 'encoded_content_id',  'user_answer',
                                                 'answered_correctly', 'question_elapsed_time', 
                                                 'question_had_explanation'],
                          dtypes = ['scalar_int', 'int', 'float', 'int', 'int', 'int', 'float', 'int'])
train_ds = tf.data.TFRecordDataset(
    tf.data.Dataset.list_files('tfrecords/whole_train/whole_train_*.tfrecords'),
    num_parallel_reads = 16
    ).map(
    parse_to_seq,
    num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).prefetch(
    tf.data.experimental.AUTOTUNE
)

for item in train_ds:
    index = data_map['next_index']
    for k in data_map['data'].keys():
        data_map['data'][k][index, :] = item[k].numpy()
    data_map['index'][item['user_id'].numpy()] = index
    data_map['next_index'] += 1
with open('data_map.pickle', 'wb') as f:
    pickle.dump(data_map, f, protocol=pickle.HIGHEST_PROTOCOL)