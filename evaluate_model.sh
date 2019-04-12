#!/usr/bin/env bash

videos_folder=$1

python main.py --video_root videos_folder --output ./output.json --model ./resnext-101-kinetics.pth --mode score --model_name resnext --model_depth 101 --resnet_shortcut B