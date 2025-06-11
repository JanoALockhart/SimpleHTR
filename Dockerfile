FROM tensorflow/tensorflow:2.4.0-gpu
WORKDIR /simpleHTR
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-get update \
    && apt-get install -y libgl1
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
