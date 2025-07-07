# Code for DHGCN
PyTorch implementation on: Dual HyperGraph Convolutional Network for EEG-Based Auditory Attention Detection.
## Introduction
We propose DHGCN, a Dual HyperGraph Convolutional Network with low parameters. It effectively models high-order spatial and temporal dependencies by constructing spatial and temporal hypergraphs from EEG signals. The network consists of a hypergraph modeling module, a dual-branch hypergraph learning (DHGL) module, and a feature fusion module. Experimental results on benchmark datasets show that DHGCN outperforms state-of-the-art AAD models, achieving superior classification accuracy while reducing the trainable parameter count by over 50%.

Jian Zhou, Yingjie Xie, Cunhang Fan, Huabin Wang, Zhao Lv, Liang Tao. DHGCN: Dual HyperGraph Convolutional Network for EEG-Based Auditory Attention Detection. In ACM MM 2025.

![image](https://github.com/nobody1219/DHGCN/blob/main/Overreview.png)

# Preprocess
- Please download the AAD dataset for training.
- The [DTU dataset](https://zenodo.org/record/1199011#.Yx6eHKRBxPa) and [MM-AAD dataset](https://dl.acm.org/doi/10.1016/j.inffus.2025.102946) are used in this paper.

# Requirements
- Python 3.12

`pip install -r requirements.txt`

# Run
- Modifying the Run Settings in `config/config.yaml`.
- Using `train.py` to train and test the model.
