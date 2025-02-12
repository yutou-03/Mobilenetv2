import numpy as np


weight = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
]

#扩充至512
weight_filled=[]
for row in weight:
    num_padding = 512 - len(row)
    expanded_row = [0] * num_padding + row
    weight_filled.append(expanded_row)

#1.将所有数据转换为二进制表示
# 转为 8 位二进制字符串
weight_in_binary = [format(x, '08b') for row in weight_filled for x in row]  #4*512
    
# 初始化切片数组，4*8的weight_sliced
weight_sliced = [['' for _ in range(8)] for _ in range(4)]  # #wbits为8

#2.按位切片
for j in range(4):
    for k in range(512):
        binary_string = weight_in_binary[j * 512 + k]  # 获取当前的二进制数
        for i in range(8):
             weight_sliced[j][i] += binary_string[7 - i] 

#3.将切片后的数据按照32bit大小重组，因为MRAM阵列存储单元大小为32bit，因此可以将512bit二进制切分为16组数据
#weight_filledin_bits = np.zeros((4, 8, 16), dtype=np.int32)
weight_filledin_bits = np.zeros((4, 8, 16),dtype='<U32')
'''for j in range(4):
    for k in range(8):
       # 分割每个切片为 16 组，每组32位
       for i in range(16):
           # 获取当前 32 位块
           block = weight_sliced[j][k*512+i * 32 : k*512+(i + 1) * 32]
           # 将当前 32 位块转换为 16 进制并存储到对应的行
           hex_value = f"32'h{int(block, 2):08X}"  
           weight_filledin_bits[i][j][i] = hex_value
'''
for j in range(4):
    for k in range(8):
        # 分割每个切片为 16 组，每组32位
        for i in range(16):
            block = weight_sliced[j][k][i*32:(i+1)*32]  # 获取当前 32 位数据块
            print(block)  
            print(len(block)) 
            print(i)
            #hex_value = f"32'h{int(block, 2):08X}"  # 将二进制块转为16进制，格式为 32'hXXXXXXXX
            weight_filledin_bits[j][k][i] = block
            print(weight_filledin_bits[j][k][i])
            
print(weight_filledin_bits)

'''
# 假设 weight_filledin_bits 已经存储了 32'h形式的字符串
for j in range(4):  # 遍历第一维
    print(f"Layer {j}:")
    for k in range(8):  # 遍历第二维
        print(f"  Row {k}: ", end="")
        for i in range(16):  # 遍历第三维
            print(f"{weight_filledin_bits[j, k, i]}", end=", ")  # 打印 32'h形式的16进制数
        print()  # 换行，开始下一行
    print()  # 换行，开始下一层
'''