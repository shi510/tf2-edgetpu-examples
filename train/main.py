import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import train.builder
import train.config
from train.input_pipeline import make_tfdataset
from train.custom_model import CustomDetectorModel
from train.custom_callback import LogCallback
from train.custom_callback import DetectorCheckpoint
from train.utils import export_tflite_graph

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping

config = train.config.config

tf.keras.backend.clear_session()

detection_model, model_config = train.builder.build(
    config['model_type'],
    config['input_shape'],
    config['num_classes'],
    config['meta_info'],
    config['checkpoint'])

tf.keras.backend.set_learning_phase(True)

batch_size = config['batch_size']
learning_rate = config['learning_rate']
num_batches = config['batch_size']

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_ds = make_tfdataset(
    config['train_file'],
    batch_size,
    config['input_shape'][:2],
    enable_aug=True)
test_ds = make_tfdataset(
    config['test_file'],
    batch_size,
    config['input_shape'][:2])

custom_model = CustomDetectorModel(
    detection_model,
    config['input_shape'],
    config['num_classes'],
    config['num_grad_accum'])
custom_model.compile(optimizer=optimizer, run_eagerly=True)

checkpoint_dir = './checkpoints/{}/best'.format(config['model_name'])
callbacks = [
    DetectorCheckpoint(detection_model, monitor='val_loss', checkpoint_dir=checkpoint_dir),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, mode='min', patience=2, min_lr=1e-5, verbose=1),
    LogCallback('./logs/'+config['model_name']),
    EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True)]

meta_info_path = './checkpoints/{}'
meta_info_path = meta_info_path.format(config['model_name'])
try:
    os.makedirs(meta_info_path, exist_ok=True)
    with open(meta_info_path+'/meta_info.config', 'w') as f:
        f.write('model{'+str(model_config)+'}')
except OSError:
    print("Error: Cannot create the directory {}".format(meta_info_path))

try:
    custom_model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=config['epoch'],
        callbacks=callbacks)
except (Exception, KeyboardInterrupt) as e:
    print()
    print('============================================')
    print('Training is canceled.')
    print(e)
    print('============================================')
    print()

export_tflite_graph(meta_info_path+'/meta_info.config', meta_info_path)


