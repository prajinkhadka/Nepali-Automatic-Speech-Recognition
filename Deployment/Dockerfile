FROM ubuntu:latest

ENV TZ=America/Los_Angeles

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update -y && \
    apt-get install -y python3-pip
RUN apt install -y git
RUN apt-get update -y
RUN apt-get install libsndfile1 -y


WORKDIR /app 
COPY requirements.txt /app
RUN pip3 install git+https://github.com/huggingface/transformers.git
RUN pip3 install -r requirements.txt 
RUN apt-get install ffmpeg -y


COPY . /app
# RUN pip3 install git+https://github.com/huggingface/transformers.git
# RUN pip3 install -r requirements.txt 
EXPOSE 5000
CMD python3 gradio_app.py
