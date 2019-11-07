import json
import os
import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_json', required=True, type=Path,
                        help='Annotation json path')
    parser.add_argument('--output_annotation_json', default='filtered_annotations.json', type=Path, required=True,
                        help='Path of the output annotation file')
    parser.add_argument('--filter_on_class', type=str, required=True,
                        help='Class to write in output json')
    parser.add_argument('--classes_for_annotation_json', type=Path, default='classes_list/class_names_list_thesis',
                        help='Class to write in output json')

    args = parser.parse_args()

    annotation_json = args.annotation_json
    output_annotation_json = args.output_annotation_json
    filter_on_class = args.filter_on_class
    classes_for_annotation_json = args.classes_for_annotation_json

    class_names = []
    with open(classes_for_annotation_json, 'r') as f:
        for row in f:
            class_names.append(row[:-1])

    output_json = {'labels': class_names}

    updated_database_videos = {}
    class_for_substitution = [ x for x in class_names if x is not filter_on_class ]

    if not class_for_substitution:
        raise Exception("Cannot substitute classes with any value")
    else:
        class_for_substitution = class_for_substitution[0] #One at random
    with open(annotation_json, 'r') as input_annotation_file:
        input_annotation_json = json.load(input_annotation_file)

        database = input_annotation_json['database']

        for entry in database.items():
            key, value = entry

            value['annotations']['original_label'] = value['annotations']['label']

            if value['annotations']['label'] != filter_on_class:
                value['annotations']['label'] = class_for_substitution
            updated_database_videos[key] = value

    output_json['database'] = updated_database_videos
    print(output_json)
    with open(output_annotation_json, 'w') as out_file:
        json.dump(output_json, out_file, indent=4)
