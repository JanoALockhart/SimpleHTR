FROM tensorflow/tensorflow:2.4.0-gpu
WORKDIR /simpleHTR
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
