from train.input_pipeline import make_tfdataset
import train.builder
import train.config
from train.custom_model import CustomDetectorModel
from train.custom_callback import LogCallback
from train.custom_callback import DetectorCheckpoint

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping

config = train.config.config

tf.keras.backend.clear_session()

detection_model = train.builder.build(
    config['model_type'],
    config['input_shape'],
    config['num_classes'],
    config['checkpoint'])

tf.keras.backend.set_learning_phase(True)

batch_size = config['batch_size']
learning_rate = config['learning_rate']
num_batches = config['batch_size']

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_ds = make_tfdataset(config['train_file'], batch_size, config['input_shape'][:2])

custom_model = CustomDetectorModel(
    detection_model, config['input_shape'], config['num_grad_accum'])
custom_model.compile(optimizer=optimizer, run_eagerly=True)

checkpoint_dir = './checkpoints/{}/best'.format(config['model_name'])
callbacks = [
    LogCallback('./logs/'+config['model_name']),
    DetectorCheckpoint(detection_model, monitor='loss', checkpoint_dir=checkpoint_dir),
    ReduceLROnPlateau(monitor='loss', factor=0.1, mode='min', patience=2, min_lr=1e-5, verbose=1),
    EarlyStopping(monitor='loss', mode='min', patience=5, restore_best_weights=True)]

custom_model.fit(train_ds, epochs=config['epoch'], callbacks=callbacks)

