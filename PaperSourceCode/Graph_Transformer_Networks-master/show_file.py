# -*- encoding: utf-8 -*-
'''
@File    :   show_file.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/12/31 15:26   guo zhenhao      1.0         None

@usage:
'''
import pickle
import numpy as np

f = open('/home/guozhenhao/Learn/DeepLearning/PaperSourceCode/Graph_Transformer_Networks-master/data/ACM/edges.pkl',
         'rb')
data = pickle.load(f, encoding='utf-8')
array = np.array(list(data))
print(array.tolist())  # show file
