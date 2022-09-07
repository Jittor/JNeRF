FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y upgrade \
    && apt-get -y install --no-install-recommends libgl1 tcl-dev tk-dev python3 python-is-python3 python3-pip python3-tk python3-dev git

COPY ./requirements.txt ./

RUN python3 -m pip install -U pip \
	&& pip3 --no-cache-dir install -r ./requirements.txt \
	&& rm ./requirements.txt
