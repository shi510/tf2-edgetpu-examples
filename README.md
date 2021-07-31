# tf2-edgetpu-examples

## Index
1. [Environments](#Environments)
2. [Build docker image](#Build-docker-image)
3. [Build your dataset](#Build-your-dataset)
4. [Convert the json file into TFRECORD](#Convert-the-json-file-into-TFRECORD)
5. [Train on your dataset](#Train-on-your-dataset)
6. [Convert saved model to tflite model](#Convert-saved-model-to-tflite-model)
7. [Compile tflite model for edgetpu](#Compile-tflite-model-for-edgetpu)
8. [How to custom your own model](#How-to-custom-your-own-model)

## Environments
(1) tensorflow v2.5.0  
(2) nvidia cuda v11.3  

## Build docker image
See docker/Dockerfile.  
```
git clone https://github.com/shi510/tf2-edgetpu-examples
docker build --tag tf2-edgetpu-examples tf2-edgetpu-examples/docker
docker run -it -v /host/dataset_dir:/root/dataset_dir --name edgetpu-test --gpus all edgetpu-test /bin/bash
```

## Build your dataset
You have image_list.json file with the format (json) as below.  
```
{
    "train/image1.jpg":{
        "detection_label":{
            "class_ids": [0, 1],
            "box_list": [[0.591, 0.436, 0.712, 0.629], [0.414, 0.464, 0.548, 0.626]]
        }
    },
    "train/image2.jpg":{
        "detection_label":{
            "class_ids": [2],
            "box_list": [[0.583, 0.539, 0.710, 0.730]]
        }
    },
  ...
}
```
The `key` is a relative path of an image.   
The `value` contains a dict which has a 'detection_label' key.  
The 'detection_label' key has class labels and bounding boxes.  
The bounding box has an order as [x_min, y_min, x_max, y_max].  

## Convert the json file into TFRECORD
Input pipeline bottleneck increases training time.  
Reading data from a large file sequentially is better than reading a lot of small sized data randomly.  
Try the command below, it generates [name.tfrecord] file from the above json file.  
```
python convert_tfrecord/main.py --root_path [path] --json_file [path] --output [name.tfrecord]
```

## Train on your dataset
Modify train/config.py.  
```
'model_name': 'your_model_name',
'num_classes': 1,
'train_file': 'path/your_train.tfrecord',
'test_file': 'path/your_test.tfrecord',
'input_shape' : [224, 224, 3],
```
If you have problems on out of GPU memory, try to decrease `batch_size` and to increase `num_grad_accum`.  
Total batch size is `batch_size` * `num_grad_accum` = 512.  
```
'batch_size' : 16,
'num_grad_accum': 32,
```
Then train your model.  
```
export PYTHONPATH=$(pwd):$(pwd)/tensorflow_models/research:$(pwd)/tensorflow_models
python train/main.py
```

## Convert saved model to tflite model
You have `checkpoints/your_model_name/saved_model` folder after training is finished.  
To convert to tflite model, Try the command below.  
It generates `your_model_name.tflite` file.  
```
python convert_tflite/main.py \
--saved_model checkpoints/your_model_name/saved_model \
--dataset your_train.tfrecord \
--image_size 224,224 \
--quant_level 2 \
--output your_model_name.tflite
```

## Compile tflite model for edgetpu
You should compile your tflite model for efficient inference on edgetpu.  
It generated `your_model_name_edgetpu.tflite` file.  
See [inference_examples/inference_on_edgetpu.py](inference_examples/inference_on_edgetpu.py).  
```
edgetpu_compiler your_model_name.tflite
```

## How to custom your own model
First change your directory to tensorflow_models/research/object_detection.  
Clone mobilenet implementations.  
```
cp models/ssd_mobilenet_v2_keras_feature_extractor.py \
models/ssd_my_model_keras_feature_extractor.py
```

Specify the options.  
The 'from_layer' is where the layer gets from.  
If a layer name is left as an empty string, constructs a new feature map
using convolution of stride 2 resulting in a spatial resolution reduction by a factor of 2.

```python
self._feature_map_layout = {
    'from_layer': ['layer_15/expansion_output', 'layer_19', 'layer_21', '', '', ''
                    ][:self._num_layers],
    'layer_depth': [-1, -1, -1, 512, 256, 128][:self._num_layers],
    'use_depthwise': self._use_depthwise,
    'use_explicit_padding': self._use_explicit_padding,
}
```

Build your backbone model that outputs multiple resolution feature maps as you specified in the above options.  
```python
def build(self, input_shape):
    # Clone existing backbone or implement your model from scratch.
    full_mobilenet_v2 = mobilenet_v2.mobilenet_v2(
        batchnorm_training=(self._is_training and not self._freeze_batchnorm),
        conv_hyperparams=(self._conv_hyperparams
                            if self._override_base_feature_extractor_hyperparams
                            else None),
        weights=None,
        use_explicit_padding=self._use_explicit_padding,
        alpha=self._depth_multiplier,
        min_depth=self._min_depth,
        include_top=False)
    y1 = full_mobilenet_v2.get_layer(name='block_3_expand').output
    y2 = full_mobilenet_v2.get_layer(name='block_6_expand').output
    y3 = full_mobilenet_v2.get_layer(name='block_10_expand').output
    # The custom model results in 3 outputs.
    self.classification_backbone = tf.keras.Model(
        inputs=full_mobilenet_v2.inputs,
        outputs=[y1, y2, y3])

    # Then generate multi-resolution feature maps based on your above configurations.
    self.feature_map_generator = (
        feature_map_generators.KerasMultiResolutionFeatureMaps(
            feature_map_layout=self._feature_map_layout,
            depth_multiplier=self._depth_multiplier,
            min_depth=self._min_depth,
            insert_1x1_conv=True,
            is_training=self._is_training,
            conv_hyperparams=self._conv_hyperparams,
            freeze_batchnorm=self._freeze_batchnorm,
            name='FeatureMaps'))
    self.built = True
```

Return multiple feature maps as you set your model's outputs.  
```python
  def _extract_features(self, preprocessed_inputs):
    """Extract features from preprocessed inputs.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    preprocessed_inputs = shape_utils.check_min_image_dim(
        33, preprocessed_inputs)

    image_features = self.classification_backbone(
        ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple))

    feature_maps = self.feature_map_generator({
        'layer_15/expansion_output': image_features[0],
        'layer_19': image_features[1],
        'layer_21': image_features[2]})

    return list(feature_maps.values())

```

Finally, you have to register your model to model builder.  
Open builders/model_builder.py.  
Import your feature extractor.  
```python
if tf_version.is_tf2():
    from object_detection.models.ssd_my_model_keras_feature_extractor import SSDMyModelKerasFeatureExtractor

if tf_version.is_tf2():
  SSD_KERAS_FEATURE_EXTRACTOR_CLASS_MAP = {
      'ssd_mobilenet_v1_keras': SSDMobileNetV1KerasFeatureExtractor,
      'ssd_mobilenet_v1_fpn_keras': SSDMobileNetV1FpnKerasFeatureExtractor,
      'ssd_mobilenet_v2_keras': SSDMobileNetV2KerasFeatureExtractor,
      'ssd_my_model_keras': SSDMyModelKerasFeatureExtractor, # <- here
```

Change directory to this repository root.  
Open train/config.py.  
Change feature_extractor to your model name.  
```
'meta_info':{
    #
    # If an empty string, it is built based on 'model_type'.
    # It is used for a custom feature extractor.
    #
    'feature_extractor': 'ssd_my_model_keras',
```
