# tf2-edgetpu-examples

Clone tensorflow model repository and checkout to 927e31aa1de2d23fd62b7b2644b67b29d658b944.  
```
git clone --depth 1 https://github.com/tensorflow/models tensorflow_models
git checkout 927e31aa1de2d23fd62b7b2644b67b29d658b944
```

Regist PYTHONPATH.  
```
export PYTHONPATH=$(pwd):$(pwd)/tensorflow_models:$(pwd)/tensorflow_models/research
```

Install python packages.  
```
pip install pillow tf_slim scipy matplotlib pyyaml dataclasses
```

Install Protoc.  
```
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.17.0/protoc-3.17.0-linux-x86_64.zip
unzip protoc-3.17.0-linux-x86_64.zip -d protoc-3.17.0-linux-x86_64
cp protoc-3.17.0-linux-x86_64/bin/protoc /usr/local/bin/
```

Install edgetpu compiler and utilities.  
```
curl -s -N https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install edgetpu-compiler wget zip
```

Generate python files from proto files.  
```
protoc --proto_path=tensorflow_models/research tensorflow_models/research/object_detection/protos/*.proto --python_out=tensorflow_models/research
```

Prepare pretrained model.  
```
mkdir -p pretrained/ssd_mobilenet_v2_320x320_coco17_tpu-8
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz
tar -xf ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz
mv ssd_mobilenet_v2_320x320_coco17_tpu-8 pretrained_models/
```
