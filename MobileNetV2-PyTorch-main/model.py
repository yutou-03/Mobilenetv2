# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Any, List

import torch
import numpy as np
torch.set_printoptions(threshold=np.inf)
from torch import Tensor
from torch import nn
from torchvision.ops.misc import Conv2dNormActivation
from mram_utils import bit_slicing_vector,bit_slicing_weight,send_to_mram

__all__ = [
    "MobileNetV2",
    "InvertedResidual",
    "mobilenet_v2",
]

mobilenet_v2_inverted_residual_cfg: List[list[int, int, int, int]] = [
    # expand_ratio, out_channels, repeated times, stride
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]

class MRAMLinear(nn.Module):
    def __init__(self, in_features, out_features, bit_width=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bit_width = bit_width
        
        # MRAM参数配置
        self.mram_addr = 0x0000  # 权重存储起始地址
        self.macro_row = 0
        self.macro_col = 0

    def forward(self, x: Tensor) -> Tensor:
        # 输入x形状: [batch_size, in_features]
        batch_size = x.shape[0]
        
        # 将输入填充到512位，并转换为位切片
        x_bits = bit_slicing_vector(x) # [bit_width, 16]
        
        # 通过Store发送输入到设备RAM
        #Load("/dev/xdma0_h2c_0", device_ram_addr, cmd, 3)
        
        # 调用Vmmul计算
        result = np.zeros((batch_size, 4), dtype=np.int32)  # 4个IP的结果
        res_addr = 0x1000
        #Vmmul("/dev/xdma0_h2c_0", (unsigned int *)vector_filledin_bits, 5 * 16, 5, 1, 0, 1, 0x0008, 5, res_addr)
        
        # 合并结果并转换为张量
        output = torch.from_numpy(result[:, :self.out_features]).float()
        return output


class MobileNetV2(nn.Module):

    def __init__(
            self,
            num_classes: int = 1000,
            width_mult: float = 1.0,
            dropout: float = 0.2,
    ) -> None:
        super(MobileNetV2, self).__init__()
        input_channels = 32
        last_channels = 1280
        classifier_channels = int(last_channels * max(1.0, width_mult))

        features: List[nn.Module] = [
            Conv2dNormActivation(3,
                                 input_channels,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1,
                                 norm_layer=nn.BatchNorm2d,
                                 activation_layer=nn.ReLU6,
                                 inplace=True,
                                 bias=False,
                                 )
        ]
        for t, c, n, s in mobilenet_v2_inverted_residual_cfg:
            output_channels = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channels, output_channels, stride, t))
                input_channels = output_channels
        features.append(
            Conv2dNormActivation(input_channels,
                                 classifier_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 norm_layer=nn.BatchNorm2d,
                                 activation_layer=nn.ReLU6,
                                 inplace=True,
                                 bias=False,
                                 ),
        )
        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(dropout, True),
            #nn.Linear(classifier_channels, num_classes),
            MRAMLinear(classifier_channels, num_classes, bit_width=8),  # 替换为自定义的MRAM全连接层
        )

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x)
        return out

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)


        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, MRAMLinear):  # 如果MRAMLinear需要特殊初始化
                nn.init.normal_(module.weight, 0, 0.01)  # 示例初始化
                nn.init.zeros_(module.bias)


class InvertedResidual(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            expand_ratio: int,
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        hidden_channels = int(round(in_channels * expand_ratio))

        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        block: List[nn.Module] = []
        if expand_ratio != 1:
            # point-wise
            block.append(
                Conv2dNormActivation(in_channels,
                                     hidden_channels,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     norm_layer=nn.BatchNorm2d,
                                     activation_layer=nn.ReLU6,
                                     inplace=True,
                                     bias=False,
                                     )
            )
        # Depth-wise + point-wise layer
        block.extend(
            [
                # Depth-wise
                Conv2dNormActivation(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=hidden_channels,
                    norm_layer=nn.BatchNorm2d,
                    activation_layer=nn.ReLU6,
                    inplace=True,
                    bias=False,
                ),
                # point-wise layer
                nn.Conv2d(hidden_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.conv = nn.Sequential(*block)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)

        if self.use_res_connect:
            out = torch.add(out, x)

        return out


def mobilenet_v2(**kwargs: Any) -> MobileNetV2:
    model = MobileNetV2(**kwargs)

    return model

