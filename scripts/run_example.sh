#!/usr/bin/env bash

DIR=$(dirname $0)
PROJECT_DIR="$(cd $DIR/..; pwd)"
RUNID=$1
echo "[RUNID]: $RUNID"

#docker build -t allrank:latest $PROJECT_DIR
docker run -e PYTHONPATH=/allrank -v $PROJECT_DIR:/allrank allrank:latest /bin/sh -c 'python allrank/main.py --config-file-name /allrank/scripts/custom_config.json --run-id bestlossfunction --job-dir /output/bestlossfunction/$RUNID'