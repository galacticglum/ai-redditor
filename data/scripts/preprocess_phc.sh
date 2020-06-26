#!/bin/bash
# 
# A script for preproceessing pornhub commennts.
# Usage: ./preprocess_phc.sh <input_file> <output_file>

if [ '$0' != '' && '$1' != '' ];  then
    python ai-redditor\preprocess.py $0 $1 --format "<|bos|><|bol|>{likes}<|eol|><|bon|>{author}<|eon|><|eq_tok|>{body}<|eos|>" --dataset-ratio 1 --shuffle --test-split-ratio 0.10
else
    echo 'Invalid arguments. See usage for documentation on arguments.'
fi