# --------------
# Code starts here
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy.stats import skew
#### Data 1
# Load the data

df = pd.read_csv(path)

# Overview of the data
df.info()

df.describe()

# Histogram showing distribution of car prices
df['price'].plot.hist(bins=12,alpha =0.5)

# Countplot of the make column
df['make'].value_counts().plot(kind='bar')


# Jointplot showing relationship between 'horsepower' and 'price' of the car
df.plot.scatter(x='horsepower',y='price',c='blue')

# Correlation heat map
f = plt.figure(figsize=(19, 15))

plt.matshow(df.corr(), fignum=f.number)

plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)

plt.yticks(range(df.shape[1]), df.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)

plt.title('Correlation Matrix', fontsize=16);

# boxplot that shows the variability of each 'body-style' with respect to the 'price'
df.boxplot(column=['price'],by=['body-style'])

#### Data 2

# Load the data
df2 = pd.read_csv(path2)

# Impute missing values with mean
df2 = df2.replace("?","NaN")

mean_imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)

df2['normalized-losses'] = mean_imputer.fit_transform(df2[['normalized-losses']])

df2['horsepower'] = mean_imputer.fit_transform(df2[['horsepower']])
# Skewness of numeric features

num_cols = df2._get_numeric_data().columns

for num_col in num_cols:
    if skew(df2[num_col].values)>1:
        print(num_col)
        df2[num_col]= np.sqrt(df2[num_col])

print(df2.head())

cat_cols = list(set(df2.columns)- set(num_cols))

# Label encode 
label_encoder  = LabelEncoder()

for cat_col in cat_cols:
        df2[cat_col]= label_encoder.fit_transform(df2[cat_col])

df2['area']=df2['height']*df2['width']

print(df2.head())


# Code ends here


