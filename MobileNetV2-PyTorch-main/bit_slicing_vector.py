import numpy as np

vector_in_binary = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
#扩充至512位
# 计算需要填充的 0 的个数
num_padding = 512 - len(vector_in_binary)
# 在数组前面填充 0
vector_in_binary = [0] * num_padding + vector_in_binary

#1.将所有数据转换为二进制表示
# 转为 8 位二进制字符串
vector_in_binary = [format(x, '08b') for x in vector_in_binary]  
# 初始化切片数组（8 个切片，每个切片是 512 位）
vector_sliced = ['' for _ in range(8)]

#2.按位切片, vector_sliced[0]由所有数据的第0位组成
for binary in vector_in_binary:
    for i in range(8):
        vector_sliced[i] += binary[7-i]  # 将当前二进制字符串的第 i 位加入到第 i 个切片，第0位指的是从右往左的第0位

# 将切片格式化为 512 位二进制字符串
#vector_sliced = [f"{slice_:0>512}" for slice_ in vector_sliced]  # 保证每个切片是 512 位
# 输出结果
'''for i, slice_ in enumerate(vector_sliced):
    print(f"vector_sliced[{i}] = {slice_}")
'''
#3.将切片后的数据按照32bit大小重组，因为MRAM阵列存储单元大小为32bit，因此可以将512bit二进制切分为16组数据
# 初始化一个 8x16 的二维数组，每个元素为 32 位数据（初始化为 0）
vector_filledin_bits = [[0] * 16 for _ in range(8)]   #inbits为8
# 将每个切片的 512 位数据分成 16 组，每组 32 位
for i in range(8):
    # 分割每个切片为 16 组，每组32位
    for j in range(16):
        # 获取当前 32 位块
        block = vector_sliced[i][j * 32 : (j + 1) * 32]
        
        # 将当前 32 位块转换为 16 进制并存储到对应的行
        hex_value = f"32'h{int(block, 2):08X}"  # 将二进制块转为16进制，格式为 32'hXXXXXXXX
        vector_filledin_bits[i][j] = hex_value
        
'''        
# 打印结果，显示每行的16个32位块
for i, row in enumerate(vector_filledin_bits):
    print(f"vector_filledin_bits[{i}] = {{")
    print(", ".join(row) + ",")  # 输出每一行的 32 位块
    print("};")
    
'''
