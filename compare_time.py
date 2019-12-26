import os
import sys
import time
import numpy as np
import pandas as pd
import torch

def numpy_ops(A, B):
    C = np.dot(A, B)
    A_mean = np.mean(A, axis=0)
    B_mean = np.mean(B, axis=0)
    A_var = np.var(A, axis=0)
    B_var = np.var(B, axis=0)
    A_normalized = (A - A_mean) / np.sqrt(A_var)
    B_normalized = (B - B_mean) / np.sqrt(B_var)
    return C

def torch_ops(A, B):
    C = torch.mm(A, B)
    A_mean = torch.mean(A, axis=0)
    B_mean = torch.mean(B, axis=0)
    A_var = torch.var(A, axis=0)
    B_var = torch.var(B, axis=0)
    A_normalized = (A - A_mean) / torch.sqrt(A_var)
    B_normalized = (B - B_mean) / torch.sqrt(B_var)
    return C

if __name__=='__main__':
    print('hello world!')

    A = np.random.randn(1000, 3000)
    B = np.random.randn(3000, 1000)

    start_t = time.time()
    for i in range(100):
        C = numpy_ops(A, B)
    print('gpu cost time: ', time.time()-start_t)

    A_tensor_cpu = torch.from_numpy(A)
    B_tensor_cpu = torch.from_numpy(B)
    start_t = time.time()
    for i in range(100):
        C_tensor_cpu = torch_ops(A_tensor_cpu, B_tensor_cpu)
    print('pytorch cpu cost time: ', time.time()-start_t)

    A_tensor_gpu = torch.from_numpy(A).to(device='cuda')
    B_tensor_gpu = torch.from_numpy(B).to(device='cuda')
    start_t = time.time()
    for i in range(100):
        C_tensor_gpu = torch_ops(A_tensor_gpu, B_tensor_gpu)
    print('pytorch gpu cost time: ', time.time()-start_t)

    print('C[500, 500] is ', C[500, 500])
    print('C_tensor_cpu[500, 500] is ', C_tensor_cpu[500, 500])
    print('C_tensor_gpu[500, 500] is ', C_tensor_gpu[500, 500])

    print('start sleep...')
    time.sleep(15)

