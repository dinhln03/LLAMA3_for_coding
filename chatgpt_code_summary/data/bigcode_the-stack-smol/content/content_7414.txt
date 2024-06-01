import numpy as np
np.random.seed(0)
import pandas as pd

df1 = pd.DataFrame(
    {
        "A": ["A0", "A1", "A2", "A3"],
        "B": ["B0", "B1", "B2", "B3"],
    },
    index=[0, 1, 2, 3]
)

df2 = pd.DataFrame(
    {
        "A": ["A4", "A5", "A6", "A7"],
        "B": ["B4", "B5", "B6", "B7"],
    },
    index=[0, 1, 2, 5]
)

df12_append = df1.append(df2)
print(df12_append)

df_concat = pd.concat([df1, df2], join="inner", ignore_index=True)
print("concatenated:", df_concat)

left = pd.DataFrame(
    {
        "key": ["K0", "K1", "K2"],
        "A": ["A0", "A1", "A2"]
    }
)

right = pd.DataFrame(
    {
        "key": ["K0", "K1", "K2"],
        "B": ["B0", "B1", "B2"]
    }
)
print(left)
print(right)
print(pd.merge(left, right, on="key"))

left = pd.DataFrame(
    {
        "A": ["A0", "A1", "A2"],
    },
    index=["K0", "K1", "K2"]
)

right = pd.DataFrame(
    {
        "B": ["B0", "B1", "B2"],
    },
    index=["K0", "K1", "K3"]
)

print(left)
print(right)
print(left.join(right, how="outer"))
