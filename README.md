# Pytorch-Lightning-Template

## Introduction

Pytorch-Lightning 是一个很好的库，或者说是Pytorch的抽象和包装。它的好处是可复用性强，易维护，逻辑清晰等。缺点也很明显，这个包需要学习和理解的内容还是挺多的，或者换句话说，很重。如果直接按照官方的模板写代码，小型project还好，如果是大型项目，有复数个需要调试验证的模型和数据集，那就不太好办，甚至更加麻烦了。经过几天的摸索和调试，我总结出了下面这样一套好用的模板，也可以说是对Pytorch-Lightning的进一步抽象。

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

## Explanation

如果对每个模型直接上plmodule，对于已有项目、别人的代码等的转换将相当耗时。另外，这样的话，你需要给每个模型都加上一些相似的代码，如`training_step`，`validation_step`。显然，这并不是我们想要的，如果真的这样做，不但不易于维护，反而可能会更加杂乱。同理，如果把每个数据集类都直接转换成pl的DataModule，也会面临相似的问题。基于这样的考量，我建议使用上述架构：

- 主目录下只放一个`main.py`文件。

- `data`和`modle`两个文件夹中放入`__init__.py`文件，做成包。这样方便导入。两个`init`文件分别是：

  - `from .data_interface import DInterface`
  - `from .model_interface import MInterface`

- 在`data_interface `中建立一个`class DInterface(pl.LightningDataModule):`用作所有数据集文件的接口。`__init__()`函数中import相应Dataset类，`setup()`进行实例化，并老老实实加入所需要的的`train_dataloader`, `val_dataloader`, `test_dataloader`函数。这些函数往往都是相似的，可以用几个输入args控制不同的部分。

- 同理，在`model_interface `中建立`class MInterface(pl.LightningModule):`类，作为模型的中间接口。`__init__()`函数中import相应模型类，然后老老实实加入`configure_optimizers`, `training_step`, `validation_step`等函数，用一个接口类控制所有模型。不同部分使用输入参数控制。

- `main.py`函数只负责：

  - 定义parser，添加parse项。
  - 选好需要的`callback`函数们。
  - 实例化`MInterface`, `DInterface`, `Trainer`。

  完事。

**需要注意的是，为了实现自动加入新model和dataset而不用更改Interface，model文件夹中的模型文件名应该使用snake case命名，如`rdn_fuse.py`，而文件中的主类则要使用对应的驼峰命名法，如`RdnFuse`**。

数据集data文件夹也是一样。

虽然对命名提出了较紧的要求，但实际上并不会影响使用，反而让你的代码结构更加清晰。希望使用时候可以注意这点，以免无法parse。