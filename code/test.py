import numpy as np

# 定义变量
num_classes = 10
length = 100
concatenated_array = np.empty((0, num_classes, length, length))

# 模拟迭代过程，每次迭代获取一个 shape 为 [bs,class,length,length] 的 numpy 向量
for i in range(10):
    new_array = np.random.rand(4, num_classes, length, length)
    concatenated_array = np.concatenate((concatenated_array, new_array), axis=0)

print("拼接后的数组形状为:", concatenated_array.shape)
