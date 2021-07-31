config = {
    'model_name': 'test',

    'checkpoint': '',

    'batch_size' : 128,
    'num_grad_accum': 4,
    'epoch' : 100,
    #
    # Shape order is [Height, Width, Channel].
    #
    'input_shape' : [224, 224, 3],
    'num_classes': 1,

    #
    # Choose one of below:
    # 1. MobileNetV2_SSD
    # 2. MobileNetV2_FPN_SSD
    # 3. EfficientDet_D0_SSD
    # 4. ResNet50V1_FPN_SSD
    #
    'model_type': 'MobileNetV2_SSD',

    'meta_info':{
        #
        # If an empty string, it is built based on 'model_type'.
        # It is used for a custom feature extractor.
        #
        'feature_extractor': '',

        'matched_threshold': 0.5,
        'unmatched_threshold': 0.5,
        'base_anchor_height': 1.0,
        'base_anchor_width': 1.0,
        'num_layers': 6,

        # bifpn is only for efficient det arch.
        'bifpn':{
            'num_iterations': 3,
            'num_filters': 64
        },
    },

    #
    # Choose one of below:
    #  1. Adam
    #  2. SGD with momentum=0.9 and nesterov=True
    #
    'optimizer' : 'Adam',

    #
    # initial learning rate.
    #
    'learning_rate' : 1e-4,

    'train_file': 'your_train.tfrecord',
    'test_file': 'your_test.tfrecord',
}
