# Pytorch-Lightning-Template: Classification

## Introduction

本目录主要提供的是classification类型的template。

不同类型的template的主要区别在于:
1. `main.py` 中callbacks的观察对象即命名方法(这里是`val_acc`)。
2. `model/model_interface.py` 中增加了对`val_acc`的计算。
3. `model`中加入了特制的`standard_net.py`，用于应对各种常见预训练模型问题。
4. `data`中的`standard_data.py`提供了分类问题中常见的数据处理方法。
5. 其他一些细节。