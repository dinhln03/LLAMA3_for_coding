# --------------
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
# code starts here
df = pd.read_csv(path)
df.head()
X = df[['ages','num_reviews','piece_count','play_star_rating','review_difficulty','star_rating','theme_name','val_star_rating','country']]
y = df['list_price']
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 6, test_size = 0.3)
# code ends here



# --------------
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
# code starts here  

cols = X_train.columns
#cols= list(X_train.columns.values)


sns.pairplot(df)

# code ends here



# --------------
# Code starts here
corr = X_train.corr()
print(corr)
X_train.drop(['play_star_rating', 'val_star_rating'], axis = 1,inplace = True) 
X_test.drop(['play_star_rating', 'val_star_rating'], axis = 1,inplace = True)
# Code ends here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import math
# Code starts here
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

def metrics(actual,pred):
    print('Mean Squared Error', mean_squared_error(actual,pred))
    print('R-Squared', r2_score(actual,pred))
metrics(y_test,y_pred)
mse = 2106.7634311857673
r2 = 0.7747160273433752
# Code ends here


# --------------
# Code starts here
residual = y_test - y_pred
plt.hist(residual)



# Code ends here


