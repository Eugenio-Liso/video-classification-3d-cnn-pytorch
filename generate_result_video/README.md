# Result Video Generation
This is a script for generating videos of classification results.  
It uses both ```output.json``` and videos as inputs and draw predicted class names in each frame.

## Requirements
* Python 3
* Pillow
* ffmpeg, ffprobe

## Usage
To generate videos based on ```../output.json```, execute the following.
```
bash annotate_videos_with_predictions.sh ../output.json <input_video_folder> <output_annotated_video_folder> ../class_names_list <frames_for_prediction>
```
The 5th parameter (frames_for_prediction) is a size of temporal unit.  

The CNN predicts class scores for a 16 frame clip (by default).  
The code averages the scores over each unit.  
The size 5 means that it averages the scores over 5 clips (i.e. 16x5 frames).  
If you use the size as 0, the scores are averaged over all clips of a video.  
