# Accuracy 

## Log file

```python
python benchmark/accuracy/benchmark_accuracy.py --output <output_json_with_predictions> --labeled_videos <file_containing_labels>
```

The labels file contains a JSON with the following structure:

```bash
{
  "name_video1": "label1"
  "name_video2": "label2"
  ...
}
```

The video name must match the one you used in the prediction phase.
The label must be a valid label (i.e. one that can be predicted).