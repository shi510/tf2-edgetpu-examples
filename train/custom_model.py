from train.utils import GradientAccumulatorModel

import tensorflow as tf


class CustomDetectorModel(GradientAccumulatorModel):

    def __init__(self, detection_model, input_shape, num_classes, num_grad_accum=1, **kargs):
        super(CustomDetectorModel, self).__init__(num_accum=num_grad_accum, **kargs)
        self.detection_model = detection_model
        self.this_input_shape = input_shape
        self.num_classes = num_classes
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.cls_loss_tracker = tf.keras.metrics.Mean(name='cls_loss')
        self.loc_loss_tracker = tf.keras.metrics.Mean(name='loc_loss')

        self.val_loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.val_cls_loss_tracker = tf.keras.metrics.Mean(name='cls_loss')
        self.val_loc_loss_tracker = tf.keras.metrics.Mean(name='loc_loss')

    def compile(self, **kargs):
        super(CustomDetectorModel, self).compile(**kargs)

    def call(self, inputs, training=False):
        image_tensors, shapes = inputs
        prediction_dict = self.detection_model.predict(image_tensors, shapes)
        return prediction_dict

    def train_step(self, data):
        imgs, labels, boxes = data
        # imgs: shape (batch_size, H, W, C)
        # labels: shape (batch_size, None, 1), None is number of instances in an image
        # boxes: shape (batch_size, None, 4), None is number of instances in an image
        labels = tf.cast(labels, tf.int32)
        labels = [tf.one_hot(x, self.num_classes) for x in labels]
        boxes = [x for x in boxes]
        batch_size = len(labels)
        shapes = tf.constant(batch_size * [self.this_input_shape], dtype=tf.int32)
        self.detection_model.provide_groundtruth(
            groundtruth_boxes_list=boxes,
            groundtruth_classes_list=labels)
        with tf.GradientTape() as tape:
            prediction_dict = self([imgs, shapes], training=True)
            losses_dict = self.detection_model.loss(prediction_dict, shapes)
            total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.accumulate_grads_and_apply(grads)
        self.loss_tracker.update_state(total_loss)
        self.cls_loss_tracker.update_state(losses_dict['Loss/classification_loss'])
        self.loc_loss_tracker.update_state(losses_dict['Loss/localization_loss'])
        return {'loss': self.loss_tracker.result(),
            self.cls_loss_tracker.name: self.cls_loss_tracker.result(),
            self.loc_loss_tracker.name: self.loc_loss_tracker.result()}


    def test_step(self, data):
        imgs, labels, boxes = data
        # imgs: shape (batch_size, H, W, C)
        # labels: shape (batch_size, None, 1), None is number of instances in an image
        # boxes: shape (batch_size, None, 4), None is number of instances in an image
        labels = tf.cast(labels, tf.int32)
        labels = [tf.one_hot(x, self.num_classes) for x in labels]
        boxes = [x for x in boxes]
        batch_size = len(labels)
        shapes = tf.constant(batch_size * [self.this_input_shape], dtype=tf.int32)
        self.detection_model.provide_groundtruth(
            groundtruth_boxes_list=boxes,
            groundtruth_classes_list=labels)
        prediction_dict = self([imgs, shapes], training=True)
        losses_dict = self.detection_model.loss(prediction_dict, shapes)
        total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
        self.val_loss_tracker.update_state(total_loss)
        self.val_cls_loss_tracker.update_state(losses_dict['Loss/classification_loss'])
        self.val_loc_loss_tracker.update_state(losses_dict['Loss/localization_loss'])
        test_logs = {
            self.val_loss_tracker.name: self.val_loss_tracker.result(),
            self.val_cls_loss_tracker.name: self.val_cls_loss_tracker.result(),
            self.val_loc_loss_tracker.name: self.val_loc_loss_tracker.result()}
        return test_logs

    @property
    def metrics(self):
        return [self.loss_tracker, self.cls_loss_tracker, self.loc_loss_tracker]
