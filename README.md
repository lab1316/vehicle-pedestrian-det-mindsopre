# Introduction

- 该仓库为三种检测模型提供训练、验证和推理支持:SSD + ResNet， SSDLite和一个仅用于检测person类的SSDLite的修改版本。
     -在COCO数据集上进行训练和验证，使用pycocotools进行评估。
  - 环境：requirements.txt

# The dataset
此存储库还使用COCO的一个子集，该子集包含仅带有person类注释的图像。要获取此数据集，请运行/utils中的`coco_subset_getter.py`。确保首先在同一个目录中创建一个与原始目录类似的空目录结构。

# Configuration
  - 在训练或推断之前，可以配置几个设置，所有设置都在/general_config目录下:

    -要使用的模型:将`general_config.py`中的`model_id`改为`constants.py`中可用的3个模型中的一个
    -要使用完整的COCO数据集或仅使用包含人物注释的图像，请更改`constants.py`中的`dataset_root`路径。数据集应该在项目的一个目录下。当使用只包含人员注释的数据集时，应该相应地设置模型。这是在`classes_config.py`中完成的，默认情况下，只有修改后的SSDLite只对人进行训练。如果需要其他设置，请更改`model_to_ids`。
    -要修改模型的锚点配置，请修改`anchor_config.py`中的`model_to_anchors`
    -运行的设备内容也可以从`general_config.py`中设置，可以是“cpu”或“cuda:0”

# Training and evaluation
-数据增强:所有模型都使用数据增强进行训练，具体来说，使用了以下内容:随机裁剪、旋转、光度失真和水平翻转。
-其他训练细节:Warm up和零权重衰减批规范和偏置层使用。完整的细节和超参数设置在params中。每个模型实验的Json文件。目录结构:/misc/experiments/model_id/params.json。
-使用pycocotools在COCO验证集上进行评估。
