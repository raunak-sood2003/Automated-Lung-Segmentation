# Automated-Lung-Segmentation
## Overview
This repository contains the code for lung segmentation using the VESSEL-12 data set (https://vessel12.grand-challenge.org/), which contains 20 lung CT volumes each consisting of 300-500 slices. Segmentation is used to remove unnecessary portions of the CT, leaving only the lung area in the image. My goal was to comapre various techniques for autonomous segmentation of lung CT, including vanilla UNETs trained on varying loss functions, UNETs with pretrained encoders and conditional generative adversarial networks.
## Model Development
The UNET model was adapted from the original paper (https://arxiv.org/abs/1505.04597) in which the input image size was 572 x 572 and the output mask was 388 x 388. 
<div>
<img src="https://github.com/raunak-sood2003/Automated-Lung-Segmentation/blob/master/Images/unet_unet15.png" width="500"/>
</div>
In this implementation, the input size was altered to account for standard DICOM image size (572 x 572), and the output mask shape matched the input size by adding a padding of one to each convolutional layer.
