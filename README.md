# PointNet: "Object Classification,Semantic Segmentation and Part Segmentation for 3D Point Clouds"
This is PyTorch implementation for the training and testing of the PointNet: a unified Deep Neural Network to carry out object classification, semantic segmentation and part segmentation on 3D Point Clouds.

This project aims to implement the PointNet architecture and conduct an in-depth study on it. The object classification, semantic
segmentation and part segmentation networks were implemented in the first step. 

In the second step, different network configurations were implemented by introducing weight initialization
and Dropout techniques to improve the model performance. Experiments on the ModelNet40
dataset for object classification, Stanford Indoor Data Scenes (S3DIS) for semantic segmentation and ShapeNet dataset for part segmentation showed improved performance of the modelin most cases. 

Further, the object classification network was generalized on the ScanObjectNN dataset to emphasize the effect on model performance on real-life scans compared to ComputerAided Design (CAD) generated point clouds. Part segmentation network was also generalized
on the ShapeNet-C dataset to gain insight into the model’s robustness to various noises. 

An ablation study on the PointNet part segmentation network shows that the modular network used in the PointNet architecture for input pose normalization has an insignificant effect on the model performance.

A detailed report of the project can be found in the assests.

# Datasets
3D Object Classification: ModelNet40, ScanObjectNN
3D Semantic Segmentation: Stanford 3D Indoor Scene Dataset (S3DIS)
3D Part Segmentation: Shapenet, Shapenet-C




