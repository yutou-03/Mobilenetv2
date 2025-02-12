import numpy as np
# 设置打印精度为8位
np.set_printoptions(precision=8, floatmode='fixed')
def quantize_to_int8(data, scale, zero_point, int8_min=-128, int8_max=127):
    # 使用 Zero Point 量化到 int8
    quantized = np.round(data / scale + zero_point).astype(np.int32)
    quantized = np.clip(quantized, int8_min, int8_max).astype(np.int8)
    return quantized

def dequantize_from_int8(quantized, scale, zero_point):
    # 使用 Zero Point 反量化到 float32
    dequantized = (quantized - zero_point) * scale
    return dequantized

# 输入 float32 数据范围
data = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float32)

# 定义 float 和 int8 的范围
float_min, float_max = -0.5, 2.5
int8_min, int8_max = -128, 127

# 计算 scale 和 zero_point
scale = (float_max - float_min) / (int8_max - int8_min)
zero_point = round(-float_min / scale) + int8_min

print(f"Scale: {scale}, Zero Point: {zero_point}")

# 量化
quantized_data = quantize_to_int8(data, scale, zero_point, int8_min, int8_max)
print("Quantized Data (int8):", quantized_data)

# 反量化
dequantized_data = dequantize_from_int8(quantized_data, scale, zero_point)
print("Dequantized Data (float32):", dequantized_data)
