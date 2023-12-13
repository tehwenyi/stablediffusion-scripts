# docker build -t dreambooth .

FROM nvcr.io/nvidia/pytorch:23.06-py3

ENV DEBIAN_FRONTEND=noninteractive

ENV cwd="/workspace/"
WORKDIR $cwd

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ENV TORCH_CUDA_ARCH_LIST="7.5 8.6"

RUN apt-get -y update \
    && apt-get -y upgrade

RUN apt-get install --no-install-recommends -y --fix-missing \
    software-properties-common \
    build-essential \
    libgl1-mesa-glx \
    git ffmpeg vim nano

RUN apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove

RUN rm -rf /var/cache/apt/archives/

### APT END ###
RUN apt-get update && apt-get install -y python3-pip
RUN python3 -m pip install --upgrade pip setuptools

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
