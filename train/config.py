config = {
    'model_name': 'test',

    'checkpoint': '',

    #
    # Shape order is [Height, Width, Channel].
    #

    #
    # Choose one of below:
    # 1. MobileNetV2_SSD
    #
    'model_type': 'MobileNetV2_SSD',

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
