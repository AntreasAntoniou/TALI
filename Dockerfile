FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

RUN apt update
RUN apt upgrade -y

RUN apt install aptitude tree -y
RUN apt install fish -y
RUN apt install wget -y

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_22.11.1-1-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p /opt/conda

SHELL ["/opt/conda/bin/conda", "run", "-n", "base", "/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0

RUN conda create -n main python=3.10 -y

SHELL ["/opt/conda/bin/conda", "run", "-n", "main", "/bin/bash", "-c"]

RUN conda install -c conda-forge mamba -y
RUN mamba install -c conda-forge starship jupyterlab black git-lfs -y
RUN mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
RUN mamba install -c conda-forge timm accelerate datasets transformers -y
RUN mamba install -c conda-forge orjson -y
RUN mamba install -c conda-forge git gh -y
RUN mamba install -c conda-forge starship -y

RUN echo y | pip install tabulate nvitop hydra_zen neptune pytorchvideo torchtyping --upgrade
RUN echo y | pip install git+https://github.com/BayesWatch/bwatchcompute@main

RUN conda init bash
RUN conda init fish

RUN apt-get install git -y

RUN git lfs install
RUN git config --global credential.helper store
RUN pip install wandb --upgrade

RUN mkdir /app/
ADD tali_wit/ /app/tali_wit
ADD setup.py /app/

RUN  /app/

RUN echo y | pip install /app/

ENTRYPOINT ["/bin/bash"]
