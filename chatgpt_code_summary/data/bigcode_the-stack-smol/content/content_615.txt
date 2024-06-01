# <a href="https://colab.research.google.com/github/couyang24/general_learning-tiffany/blob/master/Titanic/analysis/colab_titanic_main.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Need to mount Drive on or upload kaggle.json

from google.colab import drive

drive.mount("/content/drive")

# !mkdir ~/.kaggle/

# !cp drive/My\ Drive/input/kaggle.json ~/.kaggle/

# !kaggle competitions download -c titanic

# Load Package
# import numpy as np
import pandas as pd
import seaborn as sns
import featuretools

import featuretools as ft

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder,
    OrdinalEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

# Load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Save data
target = train_df[["Survived"]]
submission = test_df[["PassengerId"]]

# Join and Clean
combine = pd.concat([train_df, test_df])

# EDA
combine.info()

combine.columns

mapping = {
    "Mlle": "Miss",
    "Major": "Mr",
    "Col": "Mr",
    "Sir": "Mr",
    "Don": "Mr",
    "Mme": "Miss",
    "Jonkheer": "Mr",
    "Lady": "Mrs",
    "Capt": "Mr",
    "Countess": "Mrs",
    "Ms": "Miss",
    "Dona": "Mrs",
}

combine["Title"] = combine.Name.apply(
    lambda x: x.split(".")[0].split(",")[1].strip()
).replace(mapping)

combine.drop(["Cabin", "Ticket", "Name"], axis=1, inplace=True)

# +
# combine['Sex2'] = combine['Sex'].apply(lambda x: 0 if x=='female' else 1)

# +
# class ModifiedLabelEncoder(LabelEncoder):

#     def fit_transform(self, y, *args, **kwargs):
#         return super().fit_transform(y)

#     def transform(self, y, *args, **kwargs):
#         return super().transform(y)

# +
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder()),
    ]
)

numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")),])
# -

combine[["Sex", "Embarked", "Title"]] = categorical_transformer.fit_transform(
    combine[["Sex", "Embarked", "Title"]]
)

combine[["Age", "Fare"]] = numeric_transformer.fit_transform(combine[["Age", "Fare"]])

# +
es = ft.EntitySet(id="titanic_data")

es = es.entity_from_dataframe(
    entity_id="combine",
    dataframe=combine.drop(["Survived"], axis=1),
    variable_types={
        "Embarked": ft.variable_types.Categorical,
        "Sex": ft.variable_types.Boolean,
        "Title": ft.variable_types.Categorical,
    },
    index="PassengerId",
)

es
# -

es = es.normalize_entity(
    base_entity_id="combine", new_entity_id="Embarked", index="Embarked"
)
es = es.normalize_entity(base_entity_id="combine", new_entity_id="Sex", index="Sex")
es = es.normalize_entity(base_entity_id="combine", new_entity_id="Title", index="Title")
es = es.normalize_entity(
    base_entity_id="combine", new_entity_id="Pclass", index="Pclass"
)
es = es.normalize_entity(base_entity_id="combine", new_entity_id="Parch", index="Parch")
es = es.normalize_entity(base_entity_id="combine", new_entity_id="SibSp", index="SibSp")
es

primitives = ft.list_primitives()
pd.options.display.max_colwidth = 100
primitives[primitives["type"] == "aggregation"].head(
    primitives[primitives["type"] == "aggregation"].shape[0]
)

primitives[primitives["type"] == "transform"].head(
    primitives[primitives["type"] == "transform"].shape[0]
)

features, feature_names = ft.dfs(
    entityset=es,
    target_entity="combine",
    #                                  trans_primitives=['subtract_numeric', 'add_numeric', 'divide_numeric', 'multiply_numeric'],
    max_depth=2,
)

feature_names

len(feature_names)

features.isnull().sum()


class RemoveLowInfo(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        keep = [
            column
            for column in df.columns
            if df[column].value_counts(normalize=True).reset_index(drop=True)[0]
            < self.threshold
        ]
        return df[keep]


from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

impute_median = FunctionTransformer(lambda x: x.fillna(x.median()), validate=False)

normalize = FunctionTransformer(lambda x: (x - x.mean()) / x.std(), validate=False)

from sklearn.decomposition import PCA

transformer = Pipeline(
    [
        ("imputer", impute_median),
        ("removelowinfo", RemoveLowInfo(threshold=0.95)),
        ("scaler", normalize),
    ]
)

clean_features = transformer.fit_transform(features)

# !pip install catboost

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    VotingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb
import catboost as cgb

# +
methods = [
    ("logistic", LogisticRegression(solver="lbfgs")),
    #            ('sgd', SGDClassifier()),
    ("tree", DecisionTreeClassifier()),
    ("bag", BaggingClassifier()),
    ("xgb", xgb.XGBClassifier(max_depth=3)),
    ("lgb", lgb.LGBMClassifier(max_depth=3)),
    #            ('cgb', cgb.CatBoostClassifier(max_depth=3,silent=True)),
    ("ada", AdaBoostClassifier()),
    ("gbm", GradientBoostingClassifier()),
    ("rf", RandomForestClassifier(n_estimators=100)),
    #            ('svc', LinearSVC()),
    #            ('rbf', SVC()),
    ("nb", Pipeline([("pca", PCA()), ("gnb", GaussianNB())])),
    ("nn", MLPClassifier()),
    ("knn", KNeighborsClassifier()),
]


ensemble = VotingClassifier(
    methods,
    voting="soft",
    #         weights=[1,1,1,1,2,2,1,1],
    #         flatten_transform=True,
)

clf = Pipeline(
    [
        #          ('transformer', transformer),
        ("ensemble", ensemble),
    ]
)

clf.fit(clean_features.iloc[: train_df.shape[0], :], target)
# -

submission["Survived"] = pd.DataFrame(
    clf.predict(clean_features.iloc[train_df.shape[0] :, :])
)

print(submission.dtypes)

submission.to_csv("titanic_submission.csv", index=False)
