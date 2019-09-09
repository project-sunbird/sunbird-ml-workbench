#!/bin/bash

#opening a new terminal
#gnome-terminal --tab  #ubuntu

_cwd="$PWD"

bert_model= "uncased_L-12_H-768_A-12"
file_name=$_cwd"/inputs/uncased_L-12_H-768_A-12.zip"

echo "Starting server for Bert model $bert_model"

if [[ -f "$file_name" ]]; then
    echo "$file_name exists. Skipping download."
else
    curl -o $file_name https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
fi

echo "looking for $_cwd/inputs/$bert_model"

#curl -o $_cwd"/inputs/"$bert_model".zip" https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

if [[ -d $_cwd"/inputs/$bert_model" ]]; then
    echo "/inputs/$bert_model model directory already exists."
else
    cd $_cwd"/inputs"
    unzip "uncased_L-12_H-768_A-12.zip"
    cd ..
fi


#cd ..
#echo "The current working directory: $PWD"

#bert-serving-start -model_dir "./inputs/uncased_L-12_H-768_A-12/" -num_worker=1&


