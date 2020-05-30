#!/bin/bash
# 
# A script for preproceessing posts scraped from r/writingprompts.
# Usage: ./preprocess_writingprompts.sh <input_file> <output_file>

if [ '$0' != '' && '$1' != '' ];  then
    python ai-redditor\\preprocess.py $0 $1 title selftext
else
    echo 'Invalid arguments. See usage for documentation on arguments.'
fi