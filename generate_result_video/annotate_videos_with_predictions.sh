#!/usr/bin/env bash

output_json_file=$1
input_video_folder=$2
output_predictions_folder=$3
classes_for_annotations=$4
frames_for_prediction=$5

python generate_result_video.py --output_json ${output_json_file} --input_video_folder "${input_video_folder}" --prediction_video_folder "${output_predictions_folder}" --classes_list ${classes_for_annotations} --frames_for_prediction ${frames_for_prediction}