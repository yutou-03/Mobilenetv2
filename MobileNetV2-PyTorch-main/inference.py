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
import argparse
import json
import os

import cv2
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import Resize, ConvertImageDtype, Normalize
import numpy as np
torch.set_printoptions(threshold=np.inf)
import imgproc
import model
from utils import load_state_dict

model_names = sorted(
    name for name in model.__dict__ if name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def load_class_label(class_label_file: str, num_classes: int) -> list:
    class_label = json.load(open(class_label_file))
    class_label_list = [class_label[str(i)] for i in range(num_classes)]

    return class_label_list


def choice_device(device_type: str) -> torch.device:
    # Select model processing equipment type
    if device_type == "cuda":
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    return device


def build_model(model_arch_name: str, model_num_classes: int, device: torch.device) -> [nn.Module, nn.Module]:
    mobilenet_v2_model = model.__dict__[model_arch_name](num_classes=model_num_classes)
    mobilenet_v2_model = mobilenet_v2_model.to(device=device, memory_format=torch.channels_last)

    return mobilenet_v2_model


def preprocess_image(image_path: str, image_size: int, device: torch.device) -> torch.Tensor:
    image = cv2.imread(image_path)

    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # OpenCV convert PIL
    image = Image.fromarray(image)

    # Resize to 224
    image = Resize([image_size, image_size])(image)
    # Convert image data to pytorch format data
    tensor = imgproc.image_to_tensor(image, False, False).unsqueeze_(0)
    # 查看每个维度的大小
    print("Batch size:", tensor.shape[0])  # 第一维：批次大小
    print("Channels:", tensor.shape[1])    # 第二维：通道数
    print("Height:", tensor.shape[2])      # 第三维：高度
    print("Width:", tensor.shape[3])       # 第四维：宽度
    # 将张量转换为 NumPy 数组
    tensor_np = tensor.numpy()

    # 使用 numpy.array2string 完整地输出张量内容
    tensor_str = np.array2string(tensor_np, separator=', ', threshold=np.inf)
    with open("tensor_data.txt", "w") as f:
        f.write("Image tensor before normalization:\n")
        f.write(tensor_str)
    # Convert a tensor image to the given ``dtype`` and scale the values accordingly
    tensor = ConvertImageDtype(torch.float)(tensor)
    # Normalize a tensor image with mean and standard deviation.
    tensor = Normalize(args.model_mean_parameters, args.model_std_parameters)(tensor)

    # Transfer tensor channel image format data to CUDA device
    tensor = tensor.to(device=device, memory_format=torch.channels_last, non_blocking=True)

    return tensor


def main():
    # Get the label name corresponding to the drawing
    class_label_map = load_class_label(args.class_label_file, args.model_num_classes)

    device = choice_device(args.device_type)

    # Initialize the model
    mobilenet_v2_model = build_model(args.model_arch_name, args.model_num_classes, device)
    print(f"Build `{args.model_arch_name}` model successfully.")

    # **这里插入查看卷积层的权重代码**
    conv_weight = mobilenet_v2_model.features[0][0].weight
    print("Conv2d weight shape:", conv_weight.shape)
    print("Number of output channels:", conv_weight.shape[0])  # 输出通道数
    print("Number of input channels:", conv_weight.shape[1])   # 输入通道数
    print("Kernel size:", conv_weight.shape[2:])  # 卷积核大小 (kernel_height, kernel_width)
    
    #print("Conv2d weight:", mobilenet_v2_model.features[0][0].weight)  # Conv2d 权重
    #print("Conv2d bias:", mobilenet_v2_model.features[0][0].bias)     # Conv2d 偏置
    
    # Load model weights
    mobilenet_v2_model, _, _, _, _, _ = load_state_dict(mobilenet_v2_model, args.model_weights_path)
    print(f"Load `{args.model_arch_name}` model weights `{os.path.abspath(args.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    mobilenet_v2_model.eval()

    tensor = preprocess_image(args.image_path, args.image_size, device)

    # Inference
    with torch.no_grad():
        output = mobilenet_v2_model(tensor)

    # Calculate the five categories with the highest classification probability
    prediction_class_index = torch.topk(output, k=5).indices.squeeze(0).tolist()

    # Print classification results
    for class_index in prediction_class_index:
        prediction_class_label = class_label_map[class_index]
        prediction_class_prob = torch.softmax(output, dim=1)[0, class_index].item()
        print(f"{prediction_class_label:<75} ({prediction_class_prob * 100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_arch_name", type=str, default="mobilenet_v2")
    parser.add_argument("--model_mean_parameters", type=list, default=[0.485, 0.456, 0.406])
    parser.add_argument("--model_std_parameters", type=list, default=[0.229, 0.224, 0.225])
    parser.add_argument("--class_label_file", type=str, default="./data/ImageNet_1K_labels_map.txt")
    parser.add_argument("--model_num_classes", type=int, default=1000)
    parser.add_argument("--model_weights_path", type=str, default="./results/pretrained_models/MobileNetV2-ImageNet_1K-86ab0476.pth.tar")
    parser.add_argument("--image_path", type=str, default="./figure/n01440764_36.JPEG")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device_type", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    main()
