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

- Create a `class DInterface(pl.LightningDataModule):` in `data_interface ` to work as the interface of all different customeized Dataset files. Corresponding Dataset class is imported in the `__init__()` function. Instantiation are done in the `setup()`, and `train_dataloader`, `val_dataloader`, `test_dataloader` functions are created.

- Similarly, class `class MInterface(pl.LightningModule):` are created in `model_interface` to work as the interface of all your model files. Corresponding model class is imported in the `__init__()` function. The only things you need to modify in the interface is the functions like `configure_optimizers`, `training_step`, `validation_step` which control your own training process. One interface for all models, and the difference are handled in args.

- `main.py` is only responsible for the following tasks:

  - Define parser, add parse items. (Attention: If there are some arguments which are supposed to be controled outside, like in the command line, you can directly add a parse item in `main.py` file. For example, there is a string argument called `random_arg`, you can add `parser.add_argument('--random_arg', default='test', type=str)` to the `main.py` file.) Two `Interface` class will automatically select and pass those arguments to the corresponding model/data class.
  - Choose the needed `callback` functions, like auto-save, Early Stop, and LR Scheduler。
  - Instantiate `MInterface`, `DInterface`, `Trainer`。

Fin.

## Attention

**One thing that you need to pay attention to is, in order to let the `MInterface` and `DInterface` be able to parse your newly added models and datasets automatically by simply specify the argument `--model_name` and `--dataset`, we use snake case (like `standard_net.py`) for model/dataset file, and use the same content with camel case for class name, like `StandardNet`.**

The same is true for `data` folder.

Although this seems restricting your naming of models and datasets, but it can also make your code easier to read and understand. Please pay attention to this point to avoid parsing issues.

## Citation

If you used this template and find it helpful to your research, please consider citing our paper:

```
@article{ZHANG2023126388,
title = {Neuromorphic high-frequency 3D dancing pose estimation in dynamic environment},
journal = {Neurocomputing},
volume = {547},
pages = {126388},
year = {2023},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2023.126388},
url = {https://www.sciencedirect.com/science/article/pii/S0925231223005118},
author = {Zhongyang Zhang and Kaidong Chai and Haowen Yu and Ramzi Majaj and Francesca Walsh and Edward Wang and Upal Mahbub and Hava Siegelmann and Donghyun Kim and Tauhidur Rahman},
keywords = {Event Camera, Dynamic Vision Sensor, Neuromorphic Camera, Simulator, Dataset, Deep Learning, Human Pose Estimation, 3D Human Pose Estimation, Technology-Mediated Dancing},
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
