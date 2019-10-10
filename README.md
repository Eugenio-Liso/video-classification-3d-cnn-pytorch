# Video Classification Using 3D ResNet
This is a pytorch code for video (action) classification using 3D ResNet trained by [this code](https://github.com/kenshohara/3D-ResNets-PyTorch).  
The 3D ResNet is trained on the Kinetics dataset, which includes 400 action classes.  
This code uses videos as inputs and outputs class names and predicted class scores for each 16 frames in the score mode.  
In the feature mode, this code outputs features of 512 dims (after global average pooling) for each 16 frames.  

**Torch (Lua) version of this code is available [here](https://github.com/kenshohara/video-classification-3d-cnn).**

- https://github.com/kenshohara/3D-ResNets-PyTorch (MODELLO PER INFERENCE: https://github.com/kenshohara/video-classification-3d-cnn-pytorch) con paper https://arxiv.org/abs/1708.07632 + https://arxiv.org/abs/1711.09577

## Requirements
- Setup env (with Python 3.7.x)

  ```bash
  conda install seaborn=0.9.0 matplotlib=3.0.3 scikit-learn=0.21.2 pytorch=1.0.0 torchvision=0.2.1 cuda80=1.0 -c soumith
  conda install pip
  pip install opencv-python==3.4.5.20
  ```
- FFmpeg, FFprobe

## Preparation
* Download this code.
* Download the [pretrained model](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M?usp=sharing).  
  * ResNeXt-101 achieved the best performance in our experiments. (See [paper](https://arxiv.org/abs/1711.09577) in details.)

## Usage
```bash
# With custom classes. Resnet 101. Live predictions
python main.py \
--video_root ./videos \
--model ... \
--mode score \
--model_name resnet \
--model_depth 101 \
--resnet_shortcut B \
--type_of_prediction live \
--class_names_list classes_list/class_names_list_thesis

# Kinetics classes. Resnet 101. Live predictions
python main.py \
--video_root ./videos \
--model ... \
--mode score \
--model_name resnet \
--model_depth 101 \
--resnet_shortcut B \
--type_of_prediction live \
--class_names_list classes_list/class_names_list_kinetics
```
To visualize the classification results, use ```generate_result_video/generate_result_video.py```.

Also, take a look at the various scripts in the `benchmark` folder.