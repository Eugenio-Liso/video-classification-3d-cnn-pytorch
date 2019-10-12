# Metrics

## Generate test set labeling

We will refer to the video directory with `val_or_test_set`. Assuming a video structure like this:

```bash
val_or_test_set
├── falling
│   ├── video_id1
│   │   ├── image_00001.jpg
│   │   ├── image_00002.jpg
│   │   ├── image_00003.jpg
│   │   ├── image_00004.jpg
│   │   ├── image_00005.jpg
│   │   ├── image_00006.jpg
│   │   ├── image_00007.jpg
│   ├── video_id2
│   │   ├── image_00001.jpg
│   │   ├── image_00002.jpg
│   │   ├── image_00003.jpg
...
├── running
└── walking
```

Run:

```bash
python automatic_test_set_labeling.py \ 
--labeled_videos_output out_labels.json \
--videos_dir val_or_test_set
```

This will create a file with this structure: 

```bash
{
  "name_video1": "label1"
  "name_video2": "label2"
  ...
}
```

The video name must match the one you used in the prediction phase.

The label must be a valid label (i.e. one that can be predicted).

## Generate CSV containing the metrics

```python
python benchmark/metrics/benchmark_metrics.py \
--output <output json from prediction phase> \
--labeled_videos out_labels.json \
--classes_list <class_list file> \
--output_times <output_times from prediction phase>
--output_csv metrics_output.csv
```

The classes list file is a simple file containing a set of labels, one per row.

# Plotting the results

```python
python benchmark/metrics/plot_metrics.py \
--input_csv metrics_output.csv \
--classes_list <class_list file>
```