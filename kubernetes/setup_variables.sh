#!/bin/bash
export NEPTUNE_API_TOKEN=""
export NEPTUNE_PROJECT=""
export NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE='TRUE'

export WANDB_API_KEY=""
export WANDB_ENTITY=""
export WANDB_PROJECT=""

export EXPERIMENT_NAME=""
export HF_USERNAME=""
export HF_TOKEN=""

export TOKENIZERS_PARALLELISM=False

export CODE_DIR=/app/
export PROJECT_DIR=/app/
export EXPERIMENT_NAME_PREFIX="tali-exp-2"
export EXPERIMENTS_DIR=/experiments/
export EXPERIMENT_DIR=/experiments/
export TALI_DATASET_DIR=/tali-data/
export WIT_DATASET_DIR=/wit-data/
export MODEL_DIR=/model/

export CLUSTER_NAME=spot-gpu-cluster-1
export CLUSTER_ZONE=us-central1-c
export CLUSTER_PROJECT=tali-multi-modal

export DOCKER_IMAGE_PATH="gcr.io/tali-multi-modal/tali:latest"

