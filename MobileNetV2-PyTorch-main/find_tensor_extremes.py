import torch
import re

def find_tensor_extremes(file_path):
    try:
        # 读取文件内容
        with open(file_path, 'r') as file:
            content = file.read()

        # 使用正则表达式提取 tensor 数据
        tensor_data = re.findall(r"[-+]?[0-9]*\.?[0-9]+e?[-+]?\d*", content)

        # 转换为浮点数列表
        tensor_values = [float(value) for value in tensor_data]

        # 转换为 PyTorch tensor
        tensor = torch.tensor(tensor_values)

        # 找出最大值和最小值
        max_value = torch.max(tensor)
        min_value = torch.min(tensor)

        print(f"Maximum value: {max_value}")
        print(f"Minimum value: {min_value}")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# 调用函数
find_tensor_extremes("input_vector.txt")
