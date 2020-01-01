from tqdm import tqdm
import time

d = {'loss':0.2,'learn':0.8}
for i in tqdm(range(50), desc='第一层循环', ncols=100, postfix=d): #desc设置名称,ncols设置进度条长度.postfix以字典形式传入详细信息
    for i in tqdm(range(50), desc='第二层循环', ncols=100, postfix=d): #desc设置名称,ncols设置进度条长度.postfix以字典形式传入详细信息
        time.sleep(0.1)
        pass
print('prog ends here!')
