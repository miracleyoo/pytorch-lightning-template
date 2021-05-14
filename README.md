# Pytorch-Lightning-Template

[**Chinese Version 中文版**](./Assets/README_CN.md)

## Introduction

Pytorch-Lightning is a very convenient library. It can be seen as an abstraction and packaging of Pytorch. Its advantages are strong reusability, easy maintenance, clear logic, etc. The disadvantage is that it is too heavy and requires quite a bit of time to learn and understand. In addition, since it directly binds the model and the training code, it is not suitable for real projects with multiple model and dataset files. The same is true for the data module design. The strong coupling of things like DataLoader and custom Datasets also causes a similar problem: the same code is copied and pasted inelegantly here and there.

After much exploration and practice, I have summarized the following templates, which can also be a further abstraction of Pytorch-Lightning. In the first version, all the template content is under the root folder. However, after using it for more than a month, I found that more specified templates for different types of projects can boost coding efficiency. For example, classification and super-resolution tasks all have some fixed demand points. The project code can be implemented faster by directly modifying specialized templates, and some avoidable bugs have also been reduced. 

**Currently, since this is still a new library, there are only these two templates. However, later as I apply it to other projects, new specialized templates will also be added. If you have used this template for your tasks (such as NLP, GAN, speech recognition, etc.), you are welcome to submit a PR so that you can integrate your template into the library for more people to use. If your task is not on the list yet, starting from the `classification` template is a good choice. Since most of the underlying logic and code of the templates are the same, this can be done very quickly. **

Everyone is welcome to try this set of code styles. It is quite convenient to reuse if you are used to it, and it is not easy to fall back into the hole. A more detailed explanation and a complete guide to Pytorch-Lightning can be found in the [this article](https://zhuanlan.zhihu.com/p/353985363) Zhihu blog.

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

No installation is needed. Directly run `git clone https://github.com/miracleyoo/pytorch-lightning-template.git` to clone it to your local position. Choose your problem type like `classification`, and copy the corresponding template to your project directory.

## Explanation of Structure

- Thre are only `main.py` and `utils.py` in the root directory. The former is the entrance of the code, and the latter is a support file.

- There is a `__init__.py` file in both `data` and `modle` folder to make them into packages. In this way, the import becomes easier.

- 在`data_interface `中建立一个`class DInterface(pl.LightningDataModule):`用作所有数据集文件的接口。`__init__()`函数中import相应Dataset类，`setup()`进行实例化，并老老实实加入所需要的的`train_dataloader`, `val_dataloader`, `test_dataloader`函数。这些函数往往都是相似的，可以用几个输入args控制不同的部分。

- 同理，在`model_interface `中建立`class MInterface(pl.LightningModule):`类，作为模型的中间接口。`__init__()`函数中import相应模型类，然后老老实实加入`configure_optimizers`, `training_step`, `validation_step`等函数，用一个接口类控制所有模型。不同部分使用输入参数控制。

- `main.py` is only responsible for the following tasks:

  - 定义parser，添加parse项。（注意如果你的模型或数据集文件的`__init__`函数中有需要外部控制的变量，如一个`random_arg`，你可以直接在`main.py`的Parser中添加这样一项，如`parser.add_argument('--random_arg', default='test', type=str)`，两个`Interface`类会自动传导这些参数到你的模型或数据集类中。）
  - Choose the needed `callback` functions, like auto-save, Early Stop, and LR Scheduler。
  - Instantiate `MInterface`, `DInterface`, `Trainer`。

Fin.

**One thing that you need to pay attention to is, in order to let the `MInterface` and `DInterface` be able to parse your newly added models and datasets automatically by simply specify the argument `--model_name` and `--dataset`, we use snake case (like `standard_net.py`) for model/dataset file, and use the same content with camel case for class name, like `StandardNet`.**

The same is true for `data` folder.

Although this seems restricting your naming of models and datasets, but it can also make your code easier to read and understand. Please pay attention to this point to avoid parsing issues.