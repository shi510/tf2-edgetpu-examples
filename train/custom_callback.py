import tensorflow as tf
import numpy as np

class LogCallback(tf.keras.callbacks.Callback):

    def __init__(self, log_dir='./logs'):
        super(LogCallback, self).__init__()
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def on_train_end(self, logs=None):
        self.writer.close()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=epoch)
            self.writer.flush()

class DetectorCheckpoint(tf.keras.callbacks.Callback):

    def __init__(self, detection_model, monitor='loss', checkpoint_dir='./checkpoints'):
        super(DetectorCheckpoint, self).__init__()
        self.ckpt_dir = checkpoint_dir
        box_heads_ckpt = tf.compat.v2.train.Checkpoint(
            _prediction_heads=detection_model._box_predictor._prediction_heads)
        detection_model_ckpt = tf.compat.v2.train.Checkpoint(
                _feature_extractor=detection_model._feature_extractor,
                _box_predictor=box_heads_ckpt)
        self.ckpt = tf.compat.v2.train.Checkpoint(model=detection_model_ckpt)
        self.monitor = monitor
        self.monitor_val = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.monitor in logs:
            if self.monitor_val > logs[self.monitor]:
                self.ckpt.save(self.ckpt_dir)
                print('{} is improved from {} to {}, '
                    .format(self.monitor, self.monitor_val, logs[self.monitor]), end='')
                print('saving model to {}'.format(self.ckpt_dir))
                self.monitor_val = logs[self.monitor]
        else:
            print('{} monitor not found in logs.'.format(self.monitor))
