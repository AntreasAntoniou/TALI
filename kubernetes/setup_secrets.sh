#!/bin/bash
kubectl create secret generic $EXPERIMENT_NAME_PREFIX \
    --from-literal=NEPTUNE_API_TOKEN=$NEPTUNE_API_TOKEN --from-literal=HF_TOKEN=$HF_TOKEN
