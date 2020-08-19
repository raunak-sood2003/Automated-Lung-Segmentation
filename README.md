# Automated-Lung-Segmentation
## Overview
This repository contains the code for lung segmentation using the VESSEL-12 data set (https://vessel12.grand-challenge.org/), which contains 20 lung CT volumes each consisting of 300-500 slices. Segmentation is used to remove unnecessary portions of the CT, leaving only the lung area in the image. My goal was to comapre various techniques for autonomous segmentation of lung CT, including vanilla UNETs trained on varying loss functions, UNETs with pretrained encoders and conditional generative adversarial networks.
## Model Development
The UNET model was adapted from the original paper (https://arxiv.org/abs/1505.04597) in which the input image size was 572 x 572 and the output mask was 388 x 388. 
<p align="center">
  <img src="https://github.com/raunak-sood2003/Automated-Lung-Segmentation/blob/master/Images/unet_unet15.png" />
</p>
In this implementation, the input size was altered to account for standard DICOM image size (572 x 572), and the output mask shape matched the input size by adding a padding of one to each convolutional layer. The UNET model is split into two parts: the encoder and decoder. In the encoding phase, convolutional layers alternate with max pooling layers to downsample the image as the number of channels increase. Then in the decoding phase, transpose convolutional layers are used to upsample the encoded tensor. The grey arrows in the image represent concatenations that assist with upsampling.
## Training
The UNET model was trained on three different loss functions: binary crossentropy (BCE), mean squared error (MSE) and soft dice loss. Cross entropy and dice loss are the traditional losss functions used in segmentation tasks, although MSE has shown promising results in some studies. Additionally, the models were trained for 15 and 30 epoch with a batch size of 5 and a learning rate of 0.01. Adam optimization was used on all models.

![alt-text-1](https://github.com/raunak-sood2003/Automated-Lung-Segmentation/blob/master/Images/loss_unet15.png) ![alt-text-2](https://github.com/raunak-sood2003/Automated-Lung-Segmentation/blob/master/Images/ious_unet15.png)
<div>
<img src="https://github.com/raunak-sood2003/Automated-Lung-Segmentation/blob/master/Images/loss_unet15.png" width="500"/>
</div>
<div>
<img src="https://github.com/raunak-sood2003/Automated-Lung-Segmentation/blob/master/Images/ious_unet15.png" width="500"/>
</div>


