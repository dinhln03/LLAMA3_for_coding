import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np

from plotly.subplots import make_subplots
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

train_df = pd.read_csv("../data/titanic/train.csv")
train = train_df.copy()
family_column = train['SibSp'] + train['Parch']
train['Family'] = family_column
train = train[['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'Family', 'Embarked', 'Fare']]

# Account for missingness
train['Age'] = train['Age'].interpolate()
train['Fare'] = train['Fare'].interpolate()

train.head(5)