"""经典卷积神经网络模型库。

这里的所有网络都是**从零实现**的结构复现（LeNet-5 / AlexNet / VGG / GoogLeNet /
ResNet / DenseNet / MobileNetV2），不依赖任何在线预训练权重，保证在离线环境下也能
完整地完成「训练 -> 评估 -> 推理」闭环。

每个网络的 ``forward`` 都支持可变输入尺寸（末端使用自适应池化），因此同一个网络既可
以跑 32x32 的小数据集（如 CIFAR-10 / 演示数据集），也可以跑 224x224 的 ImageNet 风格
输入。具体建议的输入尺寸见 ``registry.py``。
"""

from __future__ import annotations

from typing import List, Optional, Type

import torch
import torch.nn as nn


# --------------------------------------------------------------------------------------
# 基础构件
# --------------------------------------------------------------------------------------
def conv_bn_relu(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                 padding: Optional[int] = None, groups: int = 1) -> nn.Sequential:
    padding = kernel_size // 2 if padding is None else padding
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def conv_relu(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
              padding: Optional[int] = None) -> nn.Sequential:
    padding = kernel_size // 2 if padding is None else padding
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.ReLU(inplace=True),
    )


# --------------------------------------------------------------------------------------
# 1. LeNet-5 (1998)
# --------------------------------------------------------------------------------------
class LeNet5(nn.Module):
    """LeCun 等人提出的卷积网络开山之作，原始任务为手写数字识别。"""

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)


# --------------------------------------------------------------------------------------
# 2. AlexNet (2012)
# --------------------------------------------------------------------------------------
class AlexNet(nn.Module):
    """2012 年 ImageNet 冠军网络，首次引入 ReLU、Dropout 与 GPU 训练。"""

    def __init__(self, num_classes: int = 1000, in_channels: int = 3, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)


# --------------------------------------------------------------------------------------
# 3. VGG (2014)
# --------------------------------------------------------------------------------------
_VGG_CONFIGS = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M",
              512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    """VGG：用 3x3 小卷积堆叠加深网络，验证了「深度」对精度的决定性作用。"""

    def __init__(self, features: nn.Sequential, num_classes: int = 1000, dropout: float = 0.5,
                 hidden: int = 4096):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)


def build_vgg(variant: str = "vgg16", num_classes: int = 1000, in_channels: int = 3,
              batch_norm: bool = False, hidden: int = 4096) -> VGG:
    layers: List[nn.Module] = []
    channels = in_channels
    for item in _VGG_CONFIGS[variant]:
        if item == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(channels, item, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(item))
            layers.append(nn.ReLU(inplace=True))
            channels = item
    return VGG(nn.Sequential(*layers), num_classes=num_classes, hidden=hidden)


# --------------------------------------------------------------------------------------
# 4. GoogLeNet / Inception-v1 (2014)
# --------------------------------------------------------------------------------------
class InceptionModule(nn.Module):
    """Inception 模块：同一层内并行使用 1x1 / 3x3 / 5x5 卷积与池化，再在通道维拼接。"""

    def __init__(self, in_channels: int, ch1x1: int, ch3x3_reduce: int, ch3x3: int,
                 ch5x5_reduce: int, ch5x5: int, pool_proj: int):
        super().__init__()
        self.branch1 = conv_relu(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            conv_relu(in_channels, ch3x3_reduce, kernel_size=1),
            conv_relu(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            conv_relu(in_channels, ch5x5_reduce, kernel_size=1),
            conv_relu(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_relu(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class InceptionAux(nn.Module):
    """GoogLeNet 的辅助分类器，训练时提供额外的梯度监督。"""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = conv_relu(in_channels, 128, kernel_size=1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.fc(self.conv(self.pool(x)))


class GoogLeNet(nn.Module):
    """GoogLeNet：22 层深，参数量却只有 VGG 的 1/12，兼顾精度与效率。"""

    def __init__(self, num_classes: int = 1000, in_channels: int = 3, aux_logits: bool = True,
                 dropout: float = 0.4):
        super().__init__()
        self.aux_logits = aux_logits
        self.conv1 = conv_relu(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = conv_relu(64, 64, kernel_size=1)
        self.conv3 = conv_relu(64, 192, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1024, num_classes)
        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)
        x = self.inception4a(x)
        aux1 = self.aux1(x) if self.training and self.aux_logits else None
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = self.aux2(x) if self.training and self.aux_logits else None
        x = self.inception4e(x)
        x = self.pool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = self.dropout(torch.flatten(x, 1))
        out = self.fc(x)
        if self.training and self.aux_logits and aux1 is not None and aux2 is not None:
            return out, aux1, aux2
        return out


# --------------------------------------------------------------------------------------
# 5. ResNet (2015)
# --------------------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet(nn.Module):
    """ResNet：通过残差连接解决深层网络的退化问题，是工业界最常用的骨干网络之一。"""

    def __init__(self, block: Type[nn.Module], layers: List[int], num_classes: int = 1000,
                 in_channels: int = 3, zero_init_residual: bool = False):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[nn.Module], planes: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


def build_resnet(variant: str = "resnet18", num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    configs = {
        "resnet18": (BasicBlock, [2, 2, 2, 2]),
        "resnet34": (BasicBlock, [3, 4, 6, 3]),
        "resnet50": (Bottleneck, [3, 4, 6, 3]),
        "resnet101": (Bottleneck, [3, 4, 23, 3]),
    }
    block, layers = configs[variant]
    return ResNet(block, layers, num_classes=num_classes, in_channels=in_channels,
                  zero_init_residual=block is Bottleneck)


# --------------------------------------------------------------------------------------
# 6. DenseNet (2016)
# --------------------------------------------------------------------------------------
class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, bn_size: int = 4, drop_rate: float = 0.0):
        super().__init__()
        inter = bn_size * growth_rate
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(inter)
        self.conv2 = nn.Conv2d(inter, growth_rate, 3, 1, 1, bias=False)
        self.drop_rate = drop_rate
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(self.relu(self.norm1(x)))
        out = self.conv2(self.relu(self.norm2(out)))
        if self.drop_rate > 0:
            out = nn.functional.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, in_channels: int, growth_rate: int, bn_size: int = 4,
                 drop_rate: float = 0.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Transition(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.pool(self.conv(self.relu(self.norm(x))))


class DenseNet(nn.Module):
    """DenseNet：每一层都与前面所有层直接相连，特征复用率高、参数效率好。"""

    def __init__(self, block_config=(6, 12, 24, 16), growth_rate: int = 32, num_classes: int = 1000,
                 in_channels: int = 3, bn_size: int = 4, drop_rate: float = 0.0,
                 init_features: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, init_features, 7, 2, 3, bias=False),
            nn.BatchNorm2d(init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        channels = init_features
        for i, num_layers in enumerate(block_config):
            self.features.add_module(f"denseblock{i + 1}", DenseBlock(num_layers, channels, growth_rate, bn_size, drop_rate))
            channels += num_layers * growth_rate
            if i != len(block_config) - 1:
                out_channels = channels // 2
                self.features.add_module(f"transition{i + 1}", Transition(channels, out_channels))
                channels = out_channels
        self.features.add_module("norm5", nn.BatchNorm2d(channels))
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.relu(x)
        x = self.avgpool(x)
        return self.classifier(torch.flatten(x, 1))


# --------------------------------------------------------------------------------------
# 7. MobileNetV2 (2018)
# --------------------------------------------------------------------------------------
class InvertedResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int):
        super().__init__()
        hidden = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(conv_bn_relu(in_channels, hidden, kernel_size=1))
        layers.extend([
            # 深度可分离卷积：depthwise + pointwise
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        return out + x if self.use_residual else out


class MobileNetV2(nn.Module):
    """MobileNetV2：倒残差 + 深度可分离卷积，面向移动端/边缘设备的高效模型。"""

    def __init__(self, num_classes: int = 1000, in_channels: int = 3, width_mult: float = 1.0,
                 dropout: float = 0.2):
        super().__init__()
        configs = [
            # t, c, n, s
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]
        input_channel = int(32 * width_mult)
        last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        features: List[nn.Module] = [conv_bn_relu(in_channels, input_channel, kernel_size=3, stride=2, padding=1)]
        for t, c, n, s in configs:
            output_channel = int(c * width_mult)
            for i in range(n):
                features.append(InvertedResidual(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        features.append(conv_bn_relu(input_channel, last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(last_channel, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(torch.flatten(x, 1))


# --------------------------------------------------------------------------------------
# 8. SqueezeNet (2016) —— 轻量级模型对照
# --------------------------------------------------------------------------------------
class Fire(nn.Module):
    def __init__(self, in_channels: int, squeeze_channels: int, expand1x1: int, expand3x3: int):
        super().__init__()
        self.squeeze = conv_relu(in_channels, squeeze_channels, kernel_size=1)
        self.expand1x1 = conv_relu(squeeze_channels, expand1x1, kernel_size=1)
        self.expand3x3 = conv_relu(squeeze_channels, expand3x3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.squeeze(x)
        return torch.cat([self.expand1x1(x), self.expand3x3(x)], dim=1)


class SqueezeNet(nn.Module):
    """SqueezeNet：用 1x1 卷积「压缩」通道，以极小参数量逼近 AlexNet 精度。"""

    def __init__(self, num_classes: int = 1000, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(3, 2, 1, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(3, 2, 1, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Conv2d(512, num_classes, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


# --------------------------------------------------------------------------------------
# 统一构建入口
# --------------------------------------------------------------------------------------
BUILDERS = {
    "lenet5": lambda num_classes, in_channels, **kw: LeNet5(num_classes, in_channels),
    "alexnet": lambda num_classes, in_channels, **kw: AlexNet(num_classes, in_channels),
    "vgg16": lambda num_classes, in_channels, **kw: build_vgg("vgg16", num_classes, in_channels,
                                                              batch_norm=kw.get("batch_norm", False)),
    "vgg11": lambda num_classes, in_channels, **kw: build_vgg("vgg11", num_classes, in_channels,
                                                              batch_norm=kw.get("batch_norm", False)),
    "googlenet": lambda num_classes, in_channels, **kw: GoogLeNet(num_classes, in_channels,
                                                                  aux_logits=kw.get("aux_logits", True)),
    "resnet18": lambda num_classes, in_channels, **kw: build_resnet("resnet18", num_classes, in_channels),
    "resnet50": lambda num_classes, in_channels, **kw: build_resnet("resnet50", num_classes, in_channels),
    "densenet121": lambda num_classes, in_channels, **kw: DenseNet((6, 12, 24, 16), 32, num_classes, in_channels),
    "mobilenetv2": lambda num_classes, in_channels, **kw: MobileNetV2(num_classes, in_channels),
    "squeezenet": lambda num_classes, in_channels, **kw: SqueezeNet(num_classes, in_channels),
}


def build_model(name: str, num_classes: int = 10, in_channels: int = 3, **kwargs) -> nn.Module:
    """按名称创建网络，例如 ``build_model('resnet18', num_classes=6)``。"""
    key = name.lower()
    if key not in BUILDERS:
        raise KeyError(f"未知的模型：{name}，可选：{sorted(BUILDERS)}")
    return BUILDERS[key](num_classes, in_channels, **kwargs)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def model_summary(model: nn.Module, input_size: int = 64, in_channels: int = 3) -> dict:
    """统计参数量、模型体积与单张图片的推理耗时。"""
    import time

    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    was_training = model.training
    model.eval()
    dummy = torch.randn(1, in_channels, input_size, input_size)
    with torch.no_grad():
        model(dummy)  # 预热
        start = time.perf_counter()
        for _ in range(3):
            model(dummy)
        latency_ms = (time.perf_counter() - start) / 3 * 1000
    if was_training:
        model.train()
    return {
        "total_params": total,
        "trainable_params": trainable,
        "params_million": round(total / 1e6, 3),
        "size_mb": round(total * 4 / 1024 / 1024, 2),
        "latency_ms": round(latency_ms, 2),
    }
