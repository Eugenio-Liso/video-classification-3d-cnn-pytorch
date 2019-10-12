# Benchmark

## Multiple plots
```bash
python benchmark/prediction_times/benchmark_times_plots.py --output_times <json_with_exec_times> --max_videos_rows 1
```

## Single plots

```bash
python benchmark/prediction_times/benchmark_times_single_plot.py --output_times <json_with_exec_times> 
```

The `json_with_exec_times` is an output of the prediction phase. It can contain more than one prediction per video. 
`max_videos_rows` specifies how many charts will be displayed in a row.

# Caveat

Unfortunately, this script will not work properly unless you have a small number of input videos.