FROM tensorflow/tensorflow:2.5.0-gpu
ENV HOME /root

# Install prerequisites
WORKDIR $HOME
RUN curl -s -N https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN apt-get update
RUN apt-get install -y wget zip git edgetpu-compiler libgl1-mesa-glx
RUN pip install pillow tf_slim tensorflow_addons scipy matplotlib pyyaml dataclasses opencv-python

# Clone main repository
RUN git clone https://github.com/shi510/tf2-edgetpu-examples

# Install protoc
RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v3.17.0/protoc-3.17.0-linux-x86_64.zip
RUN unzip protoc-3.17.0-linux-x86_64.zip -d protoc-3.17.0-linux-x86_64
RUN cp protoc-3.17.0-linux-x86_64/bin/protoc /usr/local/bin/

# Clone official tensorflow model repository
WORKDIR $HOME/tf2-edgetpu-examples
RUN git clone https://github.com/tensorflow/models tensorflow_models
RUN cd tensorflow_models && git checkout 927e31aa1de2d23fd62b7b2644b67b29d658b944

# Generate proto files
RUN protoc --proto_path=tensorflow_models/research tensorflow_models/research/object_detection/protos/*.proto --python_out=tensorflow_models/research

# Regist PYTHONPATH
ENV PROJECT_ROOT ${HOME}/tf2-edgetpu-examples
ENV PYTHONPATH ${PROJECT_ROOT}:${PROJECT_ROOT}/tensorflow_models:${PROJECT_ROOT}/tensorflow_models/research
