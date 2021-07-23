# tf2-edgetpu-examples

## 0. Environments
(1) tensorflow v2.5.0  
(2) nvidia cuda v11.3  

## 1. Build docker image
See docker/Dockerfile.  
```
git clone https://github.com/shi510/tf2-edgetpu-examples
docker build --tag tf2-edgetpu-examples tf2-edgetpu-examples/docker
docker run -it -v /host/dataset_dir:/root/dataset_dir --name edgetpu-test --gpus all edgetpu-test /bin/bash
```

## 2. Build your dataset
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

## 3. Convert the json file into TFRECORD
Input pipeline bottleneck increases training time.  
Reading data from a large file sequentially is better than reading a lot of small sized data randomly.  
Try the command below, it generates [name.tfrecord] file from the above json file.  
```
python convert_tfrecord/main.py --root_path [path] --json_file [path] --output [name.tfrecord]
```

## 4. Train on your dataset
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

## 5. Convert saved model to tflite model
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

## 6. Compile tflite model for edgetpu
You should compile your tflite model for efficient inference on edgetpu.  
It generated `your_model_name_edgetpu.tflite` file.  
See [inference_examples/inference_on_edgetpu.py](inference_examples/inference_on_edgetpu.py).  
```
edgetpu_compiler your_model_name.tflite
```
