# Pytorch-Lightning-Template

[**English Version**](../README.md)

## Introduction

Pytorch-Lightning 是一个很便利的库，它可以看作是Pytorch的抽象和包装。它的好处是可复用性强，易维护，逻辑清晰等。缺点是过重，需要学习和理解的内容较多，另外其直接将模型和训练代码强绑定的模式并不适合大型的真实项目，因为这些项目中往往有很多不同的模型需要训练测试。数据部分也是如此，DataLoader等与自定义Dataset强绑定也造成了类似的问题：同样的代码被不优雅的多次拷贝粘贴。

经过不断的摸索和调试，我总结出了下面这样一套好用的模板，也可以说是对Pytorch-Lightning的进一步抽象。初版中模板内容都在根文件夹下，但是经过一个多月的使用，我发现如果能针对不同类型的项目建立相应的模板，编码效率可以得到进一步提高。如分类任务和超分辨率任务最近都有做，但它们都有一些固定的需求点，每次直接使用经过特化的模板完成速度更快，也减少了一些可避免的bug和debug。同时，也可以添加一些仅适用于本任务的代码与文件。

**当前由于建库时间尚短，只有这两种模板。但后面随着我应用它到其他项目上，也会添加新的特化模板。如果您使用了本模板到您的任务（如NLP, GAN, 语音识别等），欢迎提出PR，以便整合您的模板到总库，方便更多人使用。如果您的任务还不在列表中，不妨从`classification`模板开始，调整配制出适合您任务的模板。由于绝大部分模板底层代码是相同的，这可以被很快完成。**

欢迎大家尝试这一套代码风格，如果用习惯的话还是相当方便复用的，也不容易半道退坑。更加详细的解释和对Pytorch-Lightning的完全攻略可以在[本篇](https://zhuanlan.zhihu.com/p/353985363)知乎博客上找到。

## File Structure

```
root-
	|-data
		|-__init__.py
		|-data_interface.py
		|-xxxdataset1.py
		|-xxxdataset2.py
		|-...
	|-model
		|-__init__.py
		|-model_interface.py
		|-xxxmodel1.py
		|-xxxmodel2.py
		|-...
	|-main.py
	|-utils.py
```

## Installation

本模板不需要安装，直接`git clone https://github.com/miracleyoo/pytorch-lightning-template.git` 到本地即可。使用时选择你需要的问题类型（如`classification`），将那个文件夹直接拷贝到你的项目文件夹中。

## Explanation

模板架构：

- 主目录下只放一个`main.py`文件和一个用于辅助的`utils.py`。

- `data`和`modle`两个文件夹中放入`__init__.py`文件，做成包。这样方便导入。两个`init`文件分别是：

  - `from .data_interface import DInterface`
  - `from .model_interface import MInterface`

- 在`data_interface `中建立一个`class DInterface(pl.LightningDataModule):`用作所有数据集文件的接口。`__init__()`函数中import相应Dataset类，`setup()`进行实例化，并老老实实加入所需要的的`train_dataloader`, `val_dataloader`, `test_dataloader`函数。这些函数往往都是相似的，可以用几个输入args控制不同的部分。

- 同理，在`model_interface `中建立`class MInterface(pl.LightningModule):`类，作为模型的中间接口。`__init__()`函数中import相应模型类，然后老老实实加入`configure_optimizers`, `training_step`, `validation_step`等函数，用一个接口类控制所有模型。不同部分使用输入参数控制。

- `main.py`函数只负责：

  - 定义parser，添加parse项。（注意如果你的模型或数据集文件的`__init__`函数中有需要外部控制的变量，如一个`random_arg`，你可以直接在`main.py`的Parser中添加这样一项，如`parser.add_argument('--random_arg', default='test', type=str)`，两个`Interface`类会自动传导这些参数到你的模型或数据集类中。）
  - 选好需要的`callback`函数们，如自动存档，Early Stop，LR Scheduler等。
  - 实例化`MInterface`, `DInterface`, `Trainer`。

完事。

**需要注意的是，为了实现自动加入新model和dataset而不用更改Interface，model文件夹中的模型文件名应该使用snake case命名，如`rdn_fuse.py`，而文件中的主类则要使用对应的驼峰命名法，如`RdnFuse`**。

数据集data文件夹也是一样。

虽然对命名提出了较紧的要求，但实际上并不会影响使用，反而让你的代码结构更加清晰。希望使用时候可以注意这点，以免无法parse。

## Citation

如果本模板对您的研究起到了一定的助力，请考虑引用我们的论文：

```
@misc{https://doi.org/10.48550/arxiv.2301.06648,
  doi = {10.48550/ARXIV.2301.06648},
  url = {https://arxiv.org/abs/2301.06648},
  author = {Zhang, Zhongyang and Chai, Kaidong and Yu, Haowen and Majaj, Ramzi and Walsh, Francesca and Wang, Edward and Mahbub, Upal and Siegelmann, Hava and Kim, Donghyun and Rahman, Tauhidur},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {YeLan: Event Camera-Based 3D Human Pose Estimation for Technology-Mediated Dancing in Challenging Environments with Comprehensive Motion-to-Event Simulator},
  publisher = {arXiv},
  year = {2023},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

@InProceedings{Zhang_2022_WACV,
    author    = {Zhang, Zhongyang and Xu, Zhiyang and Ahmed, Zia and Salekin, Asif and Rahman, Tauhidur},
    title     = {Hyperspectral Image Super-Resolution in Arbitrary Input-Output Band Settings},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
    month     = {January},
    year      = {2022},
    pages     = {749-759}
}
```
