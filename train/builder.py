import os

from object_detection.utils import config_util
from object_detection.builders import model_builder

import tensorflow.compat.v2 as tf


def build(model_type, input_shape, num_classes, checkpoint_path=''):
    print('Building model and restoring weights for fine-tuning...')

    if model_type == 'MobileNetV2_SSD':
        pipeline_config = 'pretrained_models/ssd_mobilenet_v2_320x320_coco17_tpu-8.config'
        # checkpoint_path = 'pretrained_models/ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint/ckpt-0'
    # elif model_type == 'ResNet50V1_SSD_FPN':
        # pipeline_config = 'pretrained_models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config'
        # checkpoint_path = 'pretrained_models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0'
    else:
        raise 'Unknown model_type: given {}'.format(model_type)

    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    model_config.ssd.num_classes = num_classes
    model_config.ssd.freeze_batchnorm = False
    model_config.ssd.image_resizer.fixed_shape_resizer.height = input_shape[0]
    model_config.ssd.image_resizer.fixed_shape_resizer.width = input_shape[1]
    detection_model = model_builder.build(model_config=model_config, is_training=True)

    if os.path.exists(checkpoint_path+'.index'):
        fake_box_predictor = tf.compat.v2.train.Checkpoint(
            # _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
            _prediction_heads=detection_model._box_predictor._prediction_heads,
            #    (i.e., the classification head that we *will not* restore)
            # _box_prediction_head=detection_model._box_predictor._box_prediction_head,
            )
        fake_model = tf.compat.v2.train.Checkpoint(
                _feature_extractor=detection_model._feature_extractor,
                _box_predictor=fake_box_predictor)
        ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
        ckpt.restore(checkpoint_path).expect_partial()
    elif len(checkpoint_path) != 0:
        print('You gave checkpoint path, but not exists. -> {}'.format(checkpoint_path))

    # Run model through a dummy image so that variables are created
    image, shapes = detection_model.preprocess(tf.zeros([1] + input_shape))
    prediction_dict = detection_model.predict(image, shapes)
    _ = detection_model.postprocess(prediction_dict, shapes)
    print('Weights restored!')

    return detection_model, model_config
