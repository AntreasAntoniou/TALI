#!/bin/bash
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkOTFjMTY5Zi03ZGUwLTQ4ODYtYWI0Zi1kZDEzNjlkMGI5ZjQifQ=="
export NEPTUNE_PROJECT=MachineLearningBrewery/tali-exp-1
export NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE='TRUE'

export WANDB_API_KEY="821661c6ee1657a2717093701ab76574ae1a9be0"
export WANDB_ENTITY=machinelearningbrewery
export WANDB_PROJECT=tali-exp-1

export EXPERIMENT_NAME=tali-exp-2
export HF_USERNAME="Antreas"
export HF_TOKEN=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

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

