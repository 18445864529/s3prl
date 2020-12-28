#!/bin/bash

# $1 : experiment name
# $2 : config path

DIR="/home/leo1994122701/S3PRL/benchmark/downstream/new_asr"

echo "Start running training process of E2E ASR"
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 main.py --config $2 \
    --deterministic \
    --name $1 \
    --njobs 16
    --seed 0 \
    --logdir ${DIR}/log/ \
    --ckpdir ${DIR}/ckpt/ \
    --outdir ${DIR}/result/ \
    --reserve_gpu 0 \

