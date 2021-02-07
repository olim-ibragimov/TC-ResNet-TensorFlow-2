# TC-ResNet implementation in TensorFlow 2

In this project, TC-ResNet 8 and 14 architectures are implemented in TF 2.4.1.
The goal is to provide a lightweight CNN model for Keyword Spotting with audio data in real time on mobile devices.

[Original paper](https://arxiv.org/abs/1904.03814v2)

[Author's implementation with Tensorflow](https://github.com/hyperconnect/TC-ResNet)

[Inspired from this Keras implementation](https://github.com/tranHieuDev23/TC-ResNet)

## System requirements for GPU training

- [Python 3.8](https://www.python.org/downloads/)
- [TensorFlow 2.4.1](https://www.tensorflow.org/install)
- [NVIDIA Driver 460.89](https://www.nvidia.com/en-us/drivers/results/167753/)
- [CUDA 11.0](https://developer.nvidia.com/cuda-11.0-update1-download-archive)
- [cuDNN v8.0.4 for CUDA 11.0](https://developer.nvidia.com/rdp/cudnn-archive#a-collapse804-110)


[comment]: <> (For training, testing and validation, this implementation uses [Google's Speech Command Dataset]&#40;https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html&#41;. Please download the dataset, and extract into a folder named `dataset` in the root folder of the repository.)

[comment]: <> (Run `main.py` to train the model.)

[comment]: <> (Run `live.py` to demostrate live prediction of the model in real time.)