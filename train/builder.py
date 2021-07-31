import os

from object_detection.utils import config_util
from object_detection.builders import model_builder

import tensorflow.compat.v2 as tf


def build(model_type, input_shape, num_classes, meta_info, checkpoint_path=''):
    if model_type == 'MobileNetV2_SSD':
        pipeline_config = 'train/builder_configs/ssd_mobilenet_v2.config'
    elif model_type == 'MobileNetV2_FPN_SSD':
        pipeline_config = 'train/builder_configs/ssd_mobilenet_v2_fpnlite.config'
    elif model_type == 'EfficientDet_D0_SSD':
        pipeline_config = 'train/builder_configs/ssd_efficientdet_d0.config'
    elif model_type == 'ResNet50V1_FPN_SSD':
        pipeline_config = 'train/builder_configs/ssd_resnet50_v1_fpn.config'
    else:
        raise 'Unknown model_type: given {}'.format(model_type)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    model_config.ssd.num_classes = num_classes
    model_config.ssd.freeze_batchnorm = False
    model_config.ssd.image_resizer.fixed_shape_resizer.height = input_shape[0]
    model_config.ssd.image_resizer.fixed_shape_resizer.width = input_shape[1]
    model_config.ssd.matcher.argmax_matcher.matched_threshold = meta_info['matched_threshold']
    model_config.ssd.matcher.argmax_matcher.unmatched_threshold = meta_info['unmatched_threshold']
    model_config.ssd.anchor_generator.ssd_anchor_generator.num_layers = meta_info['num_layers']
    model_config.ssd.feature_extractor.num_layers = meta_info['num_layers']
    if len(meta_info['feature_extractor']) != 0:
        model_config.ssd.feature_extractor.type = meta_info['feature_extractor']
    if model_type == 'EfficientDet_D0_SSD':
        model_config.ssd.feature_extractor.bifpn.num_iterations = meta_info['bifpn']['num_iterations']
        model_config.ssd.feature_extractor.bifpn.num_filters = meta_info['bifpn']['num_filters']
    detection_model = model_builder.build(model_config=model_config, is_training=True)

    if os.path.exists(checkpoint_path):
        ckpt = tf.train.Checkpoint(model=detection_model)
        manager = tf.train.CheckpointManager(
            ckpt, checkpoint_path, max_to_keep=1)
        if manager.latest_checkpoint is not None:
            ckpt.restore(manager.latest_checkpoint)
        else:
            print()
            print('============================================')
            print('You gave checkpoint path, but not exists. -> {}'.format(checkpoint_path))
            print('============================================')
            print()

    # Run model through a dummy image so that variables are created
    image, shapes = detection_model.preprocess(tf.zeros([1] + input_shape))
    prediction_dict = detection_model.predict(image, shapes)
    _ = detection_model.postprocess(prediction_dict, shapes)
    print('Weights restored!')

    return detection_model, model_config
