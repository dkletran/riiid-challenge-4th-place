SEQUENCE_LENGTH = 1024
BATCH_SIZE = 64
EPOCHS = 18
STEPS_PER_EPOCH = 2048
TRAIN_DATA_PATH = <gcs path to train tfrecords files>#for example 'gs://kaggle-riid-data/tfrecords/train/*.tfrecords'
VALID_DATA_PATH = <gcs path to train tfrecords files>#for example'gs://kaggle-riid-data/tfrecords/valid/*.tfrecords', set to None in case training on the whole data (no valid set)