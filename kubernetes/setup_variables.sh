#!/bin/bash
export WANDB_API_KEY="821661c6ee1657a2717093701ab76574ae1a9be0"
export WANDB_ENTITY=machinelearningbrewery
export WANDB_PROJECT=tali-debugging

export HF_USERNAME="Antreas"
export HF_TOKEN=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

export CODE_DIR=/app/
export EXPERIMENT_NAME_PREFIX="batch-size-search-v-2-0"
export EXPERIMENTS_DIR=/volume/experiments
export EXPERIMENT_DIR=/volume/experiments
export DATASET_DIR=/data/datasets/tali-wit-2-1-buckets/
export MODEL_DIR=/volume/models

export CLUSTER_NAME=spot-gpu-cluster-1
export CLUSTER_ZONE=us-central1-a
export CLUSTER_PROJECT=tali-multi-modal

export DOCKER_IMAGE_PATH="ghcr.io/antreasantoniou/tali:latest"
