#!/usr/bin/env bash

videos_folder=$1
echo "Input videos folder: ${videos_folder}"

# Take all the params in input except the first one
other_parameters="${@:2}"

echo "Additional input parameters: ${other_parameters}"

python main.py --video_root "${videos_folder}" --output ./output.json --model ./resnext-101-kinetics.pth --mode score --model_name resnext --model_depth 101 --resnet_shortcut B ${other_parameters}