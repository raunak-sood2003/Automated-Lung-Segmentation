# Automated-Lung-Segmentation
## Overview
This repository contains the code for lung segmentation using the VESSEL-12 data set (https://vessel12.grand-challenge.org/), which contains 20 lung CT volumes each consisting of 300-500 slices. Segmentation is used to remove unnecessary portions of the CT, leaving only the lung area in the image. My goal was to comapre various techniques for autonomous segmentation of lung CT, including vanilla UNETs trained on varying loss functions, UNETs with pretrained encoders and conditional generative adversarial networks. 
