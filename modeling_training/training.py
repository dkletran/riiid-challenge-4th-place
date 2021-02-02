
import typing
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import json
from transformers import WarmUp

from utils import CaptureStdoutToFile

#
# Reading content_map ids for content embeddings layers
#
with open('encoded_content_map_v2.json', 'r') as f:
    encoded_content_map = json.load(f)
    
#
# Training routines
#
def make_loss_function():
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True, 
                                              reduction=tf.keras.losses.Reduction.NONE)
    def _loss(y_true, logits):
        loss = loss_object(y_true, logits)
        return loss
    return _loss


def make_tpu_train_loop(strategy, model, optimizer = None, learning_rate = None):
    with strategy.scope():
       if optimizer is None:
          if learning_rate is None:
             learning_rate = 1e-3
          optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate, 
                                            beta_1=0.9, beta_2=0.98,
                                            epsilon=1e-9)
       training_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
       training_accuracy = tf.keras.metrics.Accuracy('training_accuracy', dtype=tf.float32)
       valid_loss = tf.keras.metrics.Mean('valid_loss', dtype=tf.float32)
       valid_accuracy = tf.keras.metrics.Accuracy('valid_accuracy', dtype=tf.float32)
       valid_auc = tf.keras.metrics.AUC(num_thresholds = 8192, 
                                              name = 'valid_auc', 
                                              dtype=tf.float32)

       loss_function = make_loss_function()

    @tf.function   
    def step_fn(inputs):
        """The computation to run on each TPU device."""
        answered_correctly = inputs['answered_correctly']
        non_padding_mask = inputs['non_padding_mask']
        label_mask = tf.cast(tf.greater_equal(answered_correctly, 0), tf.float32)
        label_mask *= non_padding_mask
        y_true = tf.clip_by_value(answered_correctly, clip_value_min=0, clip_value_max=1)
        y_true = y_true[..., tf.newaxis]
        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)
            loss = loss_function(y_true, logits)
            loss = tf.reduce_mean(loss*label_mask, axis = -1)
            loss = tf.nn.compute_average_loss(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
        training_loss.update_state(loss*strategy.num_replicas_in_sync,
                                sample_weight = tf.reduce_sum(label_mask))
        y_true = tf.reshape(y_true, (-1,))
        y_pred = tf.reshape(tf.cast(logits > 0, y_true.dtype), (-1,))
        label_mask = tf.reshape(label_mask, (-1,))
        training_accuracy.update_state(y_true, 
                                       y_pred,
                                      sample_weight=label_mask)

    @tf.function
    def train_multiple_steps(iterator, steps):
        has_data = True
        for _ in tf.range(steps):
            optional_data = iterator.get_next_as_optional()
            if not optional_data.has_value():
                has_data = False
                break
            strategy.run(step_fn, args=(optional_data.get_value(),))
        return has_data
    @tf.function
    def valid_step_fn(inputs):
        answered_correctly = inputs['answered_correctly']
        in_valid_set = inputs['in_valid_set']
        non_padding_mask = inputs['non_padding_mask']
        label_mask = tf.cast(tf.greater_equal(answered_correctly, 0), tf.float32)
        label_mask *= tf.cast(in_valid_set, tf.float32)
        label_mask *= non_padding_mask
        y_true = tf.clip_by_value(answered_correctly, clip_value_min=0, clip_value_max=1)
        y_true = y_true[..., tf.newaxis]
        inputs = {k:v for k,v in inputs.items() if k != 'in_valid_set'}
        logits = model(inputs, training=False)
        loss = loss_function(y_true, logits)
        loss = tf.reduce_mean(loss*label_mask, axis = -1)
        loss = tf.nn.compute_average_loss(loss)
        valid_loss.update_state(loss*strategy.num_replicas_in_sync,
                                sample_weight = tf.reduce_sum(label_mask))
        y_true = tf.reshape(y_true, (-1,))
        y_pred = tf.reshape(tf.cast(logits > 0, y_true.dtype), (-1,))
        label_mask = tf.reshape(label_mask, (-1,))        
        valid_accuracy.update_state(y_true, 
                               y_pred, 
                               sample_weight = label_mask)
        sigmoids = tf.reshape(tf.sigmoid(logits), (-1,))

        valid_auc.update_state(y_true, 
                               sigmoids, 
                               sample_weight = label_mask)
 
    @tf.function
    def predict_and_valid(iterator):
        while tf.constant(True):
            optional_data = iterator.get_next_as_optional()
            if not optional_data.has_value():
                break
            valid_data = optional_data.get_value()
            strategy.run(valid_step_fn, args=(valid_data,))

   
    def train_loop(train_ds, 
                   valid_ds = None, 
                   batch_size = 64,
                   steps_per_call = 128,
                   steps_per_epoch = 5500,
                   epochs = 4):

        batch_valid_ds = None
        if valid_ds is not None:
            batch_valid_ds = valid_ds.batch(
                    batch_size,
                    drop_remainder = True
                    ).prefetch(tf.data.experimental.AUTOTUNE).cache()
        if train_ds is not None:
            train_iterator = iter(strategy.experimental_distribute_dataset(
                    train_ds.batch(
                        batch_size,
                        drop_remainder = True
                        ).prefetch(tf.data.experimental.AUTOTUNE)))             
        for epoch in range(epochs):
            if train_ds is not None:
                steps_in_epoch = 0
                training_loss.reset_states()
                training_accuracy.reset_states()

                while (steps_in_epoch < steps_per_epoch):
                    steps_in_epoch += steps_per_call
                    train_multiple_steps(train_iterator,
                                          tf.convert_to_tensor(steps_per_call))
                    print('Current step: {}, training loss: {:.4f}, accuracy: {:.2f}%'.format(
                        optimizer.iterations.numpy(),
                        float(training_loss.result()),
                        float(training_accuracy.result()) * 100))
            if batch_valid_ds is not None:
                valid_loss.reset_states()
                valid_accuracy.reset_states()
                valid_auc.reset_states()
                valid_iterator = iter(strategy.experimental_distribute_dataset(
                      batch_valid_ds
                ))
                predict_and_valid(valid_iterator)
                print('Current epoch: {}, valid loss: {:.4f}, accuracy: {:.2f}%, auc: {:.4f}'.format(
                    epoch,
                    float(valid_loss.result()),
                    float(valid_accuracy.result()) * 100,
                    float(valid_auc.result())))
    return train_loop
    
#    
# Detect hardware, return appropriate distribution strategy
#
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)
from model_config import config
from training_config import *
from modeling import RiiidAnswerModel

with strategy.scope():
    tpu_model = RiiidAnswerModel(
        encoded_content_map,
        **config,
        )
#
# Functions to parse tfrecords data from gcs
#

def make_train_parse_function(sequence_length):
    train_feature_description = {
        'timestamp': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True), 
        'time_lag': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0, allow_missing=True), 
        'encoded_content_id': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True),  
        'user_answer': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True),
        'answered_correctly': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True),
        'question_elapsed_time' :tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0, allow_missing=True),
        'question_had_explanation': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True)
    }

    def parse_fn(example_proto):
        parsed_data = tf.io.parse_single_example(example_proto, train_feature_description)
        length = tf.shape(parsed_data['timestamp'])[0]
        r = tf.math.mod(length, sequence_length)
        left_pad = tf.random.uniform((), minval=0, 
                                 maxval = sequence_length - 1, 
                                 dtype=tf.int32)
        right_pad = sequence_length - r - left_pad
        if right_pad < 0:
           right_pad += sequence_length
           
        parsed_data['non_padding_mask'] = tf.ones_like(parsed_data['timestamp'], dtype = tf.float32)
        result = {
            k:tf.reshape(
                tf.pad(v, [[left_pad, right_pad]]),
                shape = (-1, sequence_length)
            ) for k,v in parsed_data.items()
        }
        return result
    return parse_fn

def make_valid_parse_function(sequence_length):

    valid_feature_description = {
        'timestamp': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True), 
        'time_lag': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0, allow_missing=True), 
        'encoded_content_id': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True),  
        'user_answer': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True),
        'answered_correctly': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True),
        'question_elapsed_time' :tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0, allow_missing=True),
        'question_had_explanation': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True),
        'in_valid_set': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True)
    }

    def parse_fn(example_proto):
        parsed_data = tf.io.parse_single_example(example_proto, valid_feature_description)
        result = parsed_data
        length = tf.shape(result['timestamp'])[0]
        to_pad = sequence_length - tf.math.mod(length, sequence_length)
        result['non_padding_mask'] = tf.ones_like(result['timestamp'], dtype = tf.float32)

        result = {
            k: tf.reshape(
                tf.pad(v, [[to_pad, 0]]),
                shape=(-1, sequence_length)
            )
            for k,v in result.items()
        }
        return result

    return parse_fn

#
# Create tf dataset for train data and valid data
#


train_ds = tf.data.TFRecordDataset(
    tf.data.Dataset.list_files(TRAIN_DATA_PATH),
    num_parallel_reads = 16
    )
train_ds = train_ds.shuffle(
    370000,
    reshuffle_each_iteration = True).repeat().map(make_train_parse_function(SEQUENCE_LENGTH),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
                        deterministic=False
                       ).unbatch().shuffle(200000)


if VALID_DATA_PATH is not None:
    valid_ds = tf.data.TFRecordDataset(
        tf.data.Dataset.list_files(VALID_DATA_PATH)
        )

    valid_ds = valid_ds.map(make_valid_parse_function(SEQUENCE_LENGTH),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE
                       ).unbatch()
else:
    valid_ds  = None

#
# Learning rate schedule: WarmUp (from huggingfaces transformers) then CosineDecay
#

decay_rate = tf.keras.experimental.CosineDecay(0.0003, 30000, .003)

learning_rate = WarmUp(0.00025, decay_rate, 4000, power=0.5)

train_loop = make_tpu_train_loop(strategy, tpu_model,
                                 learning_rate = learning_rate)
#
# Start training loop
#
with CaptureStdoutToFile('training.log'):
    train_loop(train_ds, valid_ds, 
           steps_per_call = 128, 
           batch_size = BATCH_SIZE, 
           steps_per_epoch = STEPS_PER_EPOCH, 
           epochs = EPOCHS)
#
# Saving model weights
#           
tpu_model.save_weights('weights.h5')