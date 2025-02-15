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
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# Model arch name
model_arch_name = "mobilenet_v2"
# Model normalization parameters
model_mean_parameters = [0.485, 0.456, 0.406]
model_std_parameters = [0.229, 0.224, 0.225]
# Model number class
model_num_classes = 1000
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = f"{model_arch_name}-ImageNet_1K"

if mode == "train":
    # Dataset address
    train_image_dir = "./data/MiniImageNet_1K/train"
    valid_image_dir = "./data/MiniImageNet_1K/valid"

    image_size = 224
    #batch_size = 128也改小一点
    batch_size = 16
    num_workers = 4

    # The address to load the pretrained model
    pretrained_model_weights_path = "./results/pretrained_models/MobileNetV2-ImageNet_1K-86ab0476.pth.tar"

    # Incremental training and migration training
    resume = ""

    # Total num epochs
    #epochs = 600稍微小一点
    epochs = 50

    # Loss parameters
    loss_label_smoothing = 0.1
    loss_weights = 1.0

    # Optimizer parameter
    model_lr = 0.1
    model_momentum = 0.9
    model_weight_decay = 2e-05
    model_ema_decay = 0.99998

    # Learning rate scheduler parameter
    lr_scheduler_T_0 = epochs // 4
    lr_scheduler_T_mult = 1
    lr_scheduler_eta_min = 5e-5

    # How many iterations to print the training/validate result
    train_print_frequency = 200
    valid_print_frequency = 20

if mode == "test":
    # Test data address
    test_image_dir = "./data/ImageNet_1K/ILSVRC2012_img_val"

    # Test dataloader parameters
    image_size = 224
    batch_size = 256
    num_workers = 4

    # How many iterations to print the testing result
    test_print_frequency = 20

    model_weights_path = "./results/pretrained_models/MobileNetV2-ImageNet_1K-86ab0476.pth.tar"
