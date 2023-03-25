# PointNet: "Object Classification, Semantic Segmentation and Part Segmentation for 3D Point Clouds"

Rishav Kumar Paramhans

## Introduction
This is PyTorch implementation for the training and testing of the PointNet: a unified Deep Neural Network to carry out object classification, semantic segmentation and part segmentation on 3D Point Clouds.

This project aims to implement the PointNet architecture and conduct an in-depth study on it. The object classification, semantic segmentation and part segmentation networks were implemented in the first step. 

In the second step, different network configurations were implemented by introducing weight initialization and Dropout techniques to improve the model performance. Experiments on the ModelNet40 dataset for object classification, Stanford Indoor Data Scenes (S3DIS) for semantic segmentation and ShapeNet dataset for part segmentation showed improved performance of the modelin most cases. 

Further, the object classification network was generalized on the ScanObjectNN dataset to emphasize the effect on model performance on real-life scans compared to ComputerAided Design (CAD) generated point clouds. Part segmentation network was also generalized
on the ShapeNet-C dataset to gain insight into the model’s robustness to various noises. 

An ablation study on the PointNet part segmentation network shows that the modular network used in the PointNet architecture for input pose normalization has an insignificant effect on the model performance.

A detailed report of the project can be found in the assets.


## Results
Here's some of the qualitative results from the experiments. Detailed quantitative results can be found in the project report in the assets.
![PSFigure_1](https://user-images.githubusercontent.com/65668108/203985250-840ed0a7-2374-46dc-9d5b-d2c1459614d9.png)
![PSFigure_5](https://user-images.githubusercontent.com/65668108/203985321-09aaa987-6d45-4eb3-a40b-0026e83e180b.png)
![PSFigure_10](https://user-images.githubusercontent.com/65668108/203985349-61d4b3b0-95bd-4a47-8b6e-dddd7963bebf.png)
![SSFigure5a](https://user-images.githubusercontent.com/65668108/203985392-cf66f8d6-4dc4-4f1a-aeb4-f8fe7dad1b71.png)

## Datasets
Here's the list of datasets used in this project.
* 3D Object Classification: ModelNet40, ScanObjectNN
* 3D Semantic Segmentation: Stanford 3D Indoor Scene Dataset (S3DIS)
* 3D Part Segmentation: Shapenet, Shapenet-C

## Setup

### Cuda and Python
In this Project I utilized Pytorch with Python 3.10, Cuda 11.6 and few other python libraries. However, feel free to try alternative versions or model of installation.

## Contact
For questions reagrding the project, feel free to post here or directly contact the author at rishavkrparamhans@gmail.com

