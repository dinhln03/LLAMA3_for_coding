import pandas as pd
import numpy as np


df = pd.DataFrame(np.random.randn(10,3),columns=['a','b','c'],index=list('abcdefghij'))
print(df)
df.ix[::2,0] = np.nan; df.ix[::4,1] = np.nan; df.ix[::3,2] = np.nan;

df = df.dropna(subset=['a','b'])   #mid delete rows where df['htm3']==na
bins = np.arange(-3,3,0.1)
bins = [-100,0,100]
indices = np.digitize(df.a,bins)
'''
bins代表若干连续的区间0:[-1,2),1:[2,7),2:[7,9),3:[9,10),用数组表示为：[-1,2,7,9,10]
np.digitize()函数生成一列数，对应位置的值表示参数一对应值在bins中所属区段的编号。
'''
groups = df.groupby(indices)
print('#'*20)
for i,group in groups:
    print(i,len(group))
    print(group)
print('#'*20)
print(groups.mean())
