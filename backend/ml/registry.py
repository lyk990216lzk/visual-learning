"""经典模型的元信息注册表：面向接口层暴露「有哪些模型、各自什么特点」。"""

from __future__ import annotations

from typing import Dict, List, Optional

from . import zoo

# 每个网络的简介信息，前端「模型库」页直接消费这份数据
ARCHITECTURES: List[Dict] = [
    {
        "key": "lenet5",
        "name": "LeNet-5",
        "year": 1998,
        "authors": "Yann LeCun 等",
        "paper": "Gradient-Based Learning Applied to Document Recognition",
        "family": "CNN 奠基",
        "default_input_size": 32,
        "min_input_size": 16,
        "description": "现代卷积网络的开山之作，2 个卷积层 + 3 个全连接层，最初用于手写数字识别。"
                       "结构轻量、训练极快，适合作为教学与基线对照。",
        "highlights": ["局部感受野 + 权值共享", "仅约 6 万参数", "CPU 秒级训练"],
    },
    {
        "key": "alexnet",
        "name": "AlexNet",
        "year": 2012,
        "authors": "Alex Krizhevsky 等",
        "paper": "ImageNet Classification with Deep Convolutional Neural Networks",
        "family": "深度 CNN 起点",
        "default_input_size": 224,
        "min_input_size": 32,
        "description": "2012 年 ImageNet 冠军，Top-5 错误率从 26% 降到 15%，引爆深度学习浪潮。"
                       "首次系统性使用 ReLU、Dropout 与双 GPU 训练。",
        "highlights": ["ReLU 激活", "Dropout 抑制过拟合", "5 卷积 + 3 全连接"],
    },
    {
        "key": "vgg16",
        "name": "VGG-16",
        "year": 2014,
        "authors": "Karen Simonyan, Andrew Zisserman",
        "paper": "Very Deep Convolutional Networks for Large-Scale Image Recognition",
        "family": "深度堆叠",
        "default_input_size": 224,
        "min_input_size": 32,
        "description": "全部使用 3x3 小卷积核堆叠到 16 层，结构规整统一，是迁移学习中最经典的骨干网络之一。",
        "highlights": ["3x3 小卷积堆叠", "13 卷积 + 3 全连接", "参数量约 138M"],
    },
    {
        "key": "vgg11",
        "name": "VGG-11",
        "year": 2014,
        "authors": "Karen Simonyan, Andrew Zisserman",
        "paper": "Very Deep Convolutional Networks for Large-Scale Image Recognition",
        "family": "深度堆叠",
        "default_input_size": 224,
        "min_input_size": 32,
        "description": "VGG 系列的浅版配置，只有 8 个卷积层，训练速度明显快于 VGG-16，用于对比网络深度带来的收益。",
        "highlights": ["VGG 家族最浅配置", "便于做深度消融对比"],
    },
    {
        "key": "googlenet",
        "name": "GoogLeNet (Inception-v1)",
        "year": 2014,
        "authors": "Christian Szegedy 等",
        "paper": "Going Deeper with Convolutions",
        "family": "多尺度并行",
        "default_input_size": 224,
        "min_input_size": 75,
        "description": "2014 年 ImageNet 冠军，提出 Inception 模块与 1x1 卷积降维，"
                       "22 层深度但参数量仅约 6.8M，远小于 VGG-16。",
        "highlights": ["Inception 多分支", "1x1 卷积降维", "辅助分类器"],
    },
    {
        "key": "resnet18",
        "name": "ResNet-18",
        "year": 2015,
        "authors": "Kaiming He 等",
        "paper": "Deep Residual Learning for Image Recognition",
        "family": "残差学习",
        "default_input_size": 224,
        "min_input_size": 32,
        "description": "残差学习的入门配置（4 个阶段各 2 个 BasicBlock）。"
                       "在精度、速度、参数量之间平衡良好，是教学实验与工程落地的高频选择。",
        "highlights": ["残差连接", "参数量约 11M", "训练稳定收敛快"],
    },
    {
        "key": "resnet50",
        "name": "ResNet-50",
        "year": 2015,
        "authors": "Kaiming He 等",
        "paper": "Deep Residual Learning for Image Recognition",
        "family": "残差学习",
        "default_input_size": 224,
        "min_input_size": 32,
        "description": "使用 Bottleneck 结构的 50 层残差网络，是检测、分割、识别等任务的通用骨干网络。",
        "highlights": ["Bottleneck 结构", "参数量约 25M", "工业界事实标准"],
    },
    {
        "key": "densenet121",
        "name": "DenseNet-121",
        "year": 2016,
        "authors": "Gao Huang 等",
        "paper": "Densely Connected Convolutional Networks",
        "family": "密集连接",
        "default_input_size": 224,
        "min_input_size": 32,
        "description": "每层都直接连接前面所有层的输出，特征复用率极高，相同精度下参数量比 ResNet 更少。",
        "highlights": ["密集连接", "缓解梯度消失", "特征复用"],
    },
    {
        "key": "mobilenetv2",
        "name": "MobileNetV2",
        "year": 2018,
        "authors": "Mark Sandler 等",
        "paper": "MobileNetV2: Inverted Residuals and Linear Bottlenecks",
        "family": "轻量化",
        "default_input_size": 224,
        "min_input_size": 32,
        "description": "面向移动端的轻量网络，采用倒残差结构与深度可分离卷积，参数量约 3.5M，推理速度极快。",
        "highlights": ["深度可分离卷积", "倒残差结构", "移动端部署友好"],
    },
    {
        "key": "squeezenet",
        "name": "SqueezeNet",
        "year": 2016,
        "authors": "Forrest Iandola 等",
        "paper": "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters",
        "family": "轻量化",
        "default_input_size": 224,
        "min_input_size": 32,
        "description": "用 Fire 模块（squeeze + expand）把参数量压到 1.2M，精度却接近 AlexNet，是模型压缩方向的经典工作。",
        "highlights": ["Fire 模块", "参数量 1.2M", "模型压缩经典"],
    },
]

# 当前可直接训练和推理的经典分类网络。
for _architecture in ARCHITECTURES:
    _architecture.setdefault("task", "classification")
    _architecture.setdefault("implemented", True)

# 目标检测与语义分割模型先纳入统一模型目录。具体训练/推理适配器接入后再切换为 True。
ARCHITECTURES.extend([
    {
        "key": "yolov3",
        "name": "YOLOv3",
        "year": 2018,
        "authors": "Joseph Redmon, Ali Farhadi",
        "paper": "YOLOv3: An Incremental Improvement",
        "family": "单阶段检测",
        "task": "detection",
        "implemented": True,
        "default_input_size": 416,
        "min_input_size": 320,
        "description": "采用 Darknet-53 骨干和多尺度预测，在速度与精度之间取得良好平衡，是实时目标检测的代表性模型。",
        "highlights": ["Darknet-53 骨干", "多尺度预测", "实时检测"],
    },
    {
        "key": "yolov5",
        "name": "YOLOv5",
        "year": 2020,
        "authors": "Glenn Jocher 等",
        "paper": "YOLOv5",
        "family": "单阶段检测",
        "task": "detection",
        "implemented": True,
        "default_input_size": 640,
        "min_input_size": 320,
        "description": "以 PyTorch 工程化实现和易部署著称，提供多种规模配置，覆盖从边缘设备到服务器的检测场景。",
        "highlights": ["PyTorch 实现", "多规格模型", "部署生态完善"],
    },
    {
        "key": "yolov8",
        "name": "YOLOv8",
        "year": 2023,
        "authors": "Ultralytics",
        "paper": "Ultralytics YOLOv8 Documentation",
        "family": "单阶段检测",
        "task": "detection",
        "implemented": True,
        "default_input_size": 640,
        "min_input_size": 320,
        "description": "采用 Anchor-free 检测头与解耦结构，并统一支持检测、实例分割和姿态估计等视觉任务。",
        "highlights": ["Anchor-free", "解耦检测头", "多任务支持"],
    },
    {
        "key": "yolov10",
        "name": "YOLOv10",
        "year": 2024,
        "authors": "Ao Wang 等",
        "paper": "YOLOv10: Real-Time End-to-End Object Detection",
        "family": "端到端检测",
        "task": "detection",
        "implemented": True,
        "default_input_size": 640,
        "min_input_size": 320,
        "description": "通过一致的双重分配策略去除推理阶段 NMS，降低延迟并提升端到端检测效率。",
        "highlights": ["NMS-free", "一致双重分配", "高效推理"],
    },
    {
        "key": "yolov11",
        "name": "YOLOv11",
        "year": 2024,
        "authors": "Ultralytics",
        "paper": "Ultralytics YOLO11 Documentation",
        "family": "单阶段检测",
        "task": "detection",
        "implemented": True,
        "default_input_size": 640,
        "min_input_size": 320,
        "description": "在 YOLOv8 基础上优化骨干与检测头结构，进一步提升精度、速度和参数效率。",
        "highlights": ["结构效率优化", "实时检测", "多任务统一"],
    },
    {
        "key": "fcn",
        "name": "FCN",
        "year": 2015,
        "authors": "Jonathan Long 等",
        "paper": "Fully Convolutional Networks for Semantic Segmentation",
        "family": "全卷积网络",
        "task": "segmentation",
        "implemented": True,
        "default_input_size": 512,
        "min_input_size": 256,
        "description": "首个将卷积网络端到端用于语义分割的经典方法，用全卷积层替换分类网络末端并恢复像素级预测。",
        "highlights": ["全卷积结构", "端到端分割", "像素级分类"],
    },
    {
        "key": "unet",
        "name": "U-Net",
        "year": 2015,
        "authors": "Olaf Ronneberger 等",
        "paper": "U-Net: Convolutional Networks for Biomedical Image Segmentation",
        "family": "编码器-解码器",
        "task": "segmentation",
        "implemented": True,
        "default_input_size": 512,
        "min_input_size": 256,
        "description": "采用对称编码器-解码器和跳层连接，兼顾高层语义与精细边界，是小样本分割的经典基线。",
        "highlights": ["U 型结构", "跳层连接", "边界细节清晰"],
    },
    {
        "key": "deeplabv3",
        "name": "DeepLabv3",
        "year": 2017,
        "authors": "Liang-Chieh Chen 等",
        "paper": "Rethinking Atrous Convolution for Semantic Image Segmentation",
        "family": "空洞卷积",
        "task": "segmentation",
        "implemented": True,
        "default_input_size": 512,
        "min_input_size": 256,
        "description": "使用空洞卷积与 ASPP 聚合多尺度上下文，在保持分辨率的同时扩大网络感受野。",
        "highlights": ["空洞卷积", "ASPP", "多尺度上下文"],
    },
])

_ARCH_INDEX = {item["key"]: item for item in ARCHITECTURES}
_SUMMARY_CACHE: Dict[tuple, Dict] = {}


def list_architectures() -> List[Dict]:
    return [dict(item) for item in ARCHITECTURES]


def get_architecture(key: str) -> Optional[Dict]:
    item = _ARCH_INDEX.get((key or "").lower())
    return dict(item) if item else None


def architecture_keys() -> List[str]:
    return list(_ARCH_INDEX.keys())


def implemented_architecture_keys() -> List[str]:
    """仅返回已经接入本系统训练与推理流程的网络。"""
    return [item["key"] for item in ARCHITECTURES if item.get("implemented")]


def architecture_summary(key: str, num_classes: int = 10, input_size: Optional[int] = None) -> Dict:
    """返回某个网络在指定类别数下的结构统计（参数量 / 体积 / CPU 推理耗时），结果带缓存。"""
    meta = _ARCH_INDEX.get((key or "").lower())
    if meta is None:
        raise KeyError(f"未知的模型：{key}")
    size = int(input_size or meta["default_input_size"])
    cache_key = (meta["key"], num_classes, size)
    if cache_key not in _SUMMARY_CACHE:
        model = zoo.build_model(meta["key"], num_classes=num_classes)
        summary = zoo.model_summary(model, input_size=size)
        summary["layers"] = sum(1 for _ in model.modules()) - 1
        _SUMMARY_CACHE[cache_key] = summary
    result = dict(_SUMMARY_CACHE[cache_key])
    result.update({
        "key": meta["key"],
        "name": meta["name"],
        "year": meta["year"],
        "family": meta["family"],
    })
    return result
