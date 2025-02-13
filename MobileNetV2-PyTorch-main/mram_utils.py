import numpy as np

def bit_slicing_vector(vector_in_binary):
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
            
def bit_slicing_weight(weight):
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
    return weight_filledin_bits

def send_to_mram(weight_filledin_bits, base_addr=0x0000):
    #将位切片发送到MRAM
    #需要调整逻辑
    '''
    Send("/dev/xdma0_h2c_0", 0, 0, 0, 0, (int *)(weight_filledin_bits[0]), 8 * 16)
    Send("/dev/xdma0_h2c_0", 0, 0, 1, 0, (int *)(weight_filledin_bits[1]), 8 * 16)
    Send("/dev/xdma0_h2c_0", 0, 0, 2, 0, (int *)(weight_filledin_bits[2]), 8 * 16)
    Send("/dev/xdma0_h2c_0", 0, 0, 3, 0, (int *)(weight_filledin_bits[3]), 8 * 16)
    Send("/dev/xdma0_h2c_0", 0, 1, 0, 0, (int *)(weight_filledin_bits[4]), 8 * 16)
    Send("/dev/xdma0_h2c_0", 0, 1, 1, 0, (int *)(weight_filledin_bits[5]), 8 * 16)
    Send("/dev/xdma0_h2c_0", 0, 1, 2, 0, (int *)(weight_filledin_bits[6]), 8 * 16)
    Send("/dev/xdma0_h2c_0", 0, 1, 3, 0, (int *)(weight_filledin_bits[7]), 8 * 16)
    '''
    #test   
