#!/bin/bash
set -euo pipefail

build_tag=$1
name="ml-workbench"
node=$2
org=$3

docker build -f ./Dockerfile --build-arg commit_hash=$(git rev-parse --short HEAD) -t ${org}/${name}:${build_tag}-build . 
echo {\"image_name\" : \"${name}\", \"image_tag\" : \"${build_tag}\", \"node_name\" : \"$node\"} > metadata.json
