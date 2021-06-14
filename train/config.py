config = {
    'model_name': 'test',

    'checkpoint': '',

    'batch_size' : 16,
    'num_grad_accum': 32,
    'epoch' : 40,
    'input_shape' : [512, 512, 3],
    'num_classes': 1,

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
    'test_files': 'your_test.tfrecord',
}
