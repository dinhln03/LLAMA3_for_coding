# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.0.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] {"_uuid": "6f06de1b48e35853f80eb1f3384baae8f8536b3c"}
# <h1><center><font size="6">Santander EDA, PCA and Light GBM Classification Model</font></center></h1>
#
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Another_new_Santander_bank_-_geograph.org.uk_-_1710962.jpg/640px-Another_new_Santander_bank_-_geograph.org.uk_-_1710962.jpg"></img>
#
# <br>
# <b>
# In this challenge, Santander invites Kagglers to help them identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. The data provided for this competition has the same structure as the real data they have available to solve this problem. 
# The data is anonimyzed, each row containing 200 numerical values identified just with a number.</b>
#
# <b>Inspired by Jiwei Liu's Kernel. I added Data Augmentation Segment to my kernel</b>
#
# <pre>
# <a id='0'><b>Content</b></a>
# - <a href='#1'><b>Import the Data</b></a>
# - <a href='#11'><b>Data Exploration</b></a>  
# - <a href='#2'><b>Check for the missing values</b></a>  
# - <a href='#3'><b>Visualizing the Satendar Customer Transactions Data</b></a>   
#  - <a href='#31'><b>Check for Class Imbalance</b></a>   
#  - <a href='#32'><b>Distribution of Mean and Standard Deviation</b></a>   
#  - <a href='#33'><b>Distribution of Skewness</b></a>   
#  - <a href='#34'><b>Distribution of Kurtosis</b></a>   
# - <a href='#4'><b>Principal Component Analysis</b></a>
#  - <a href='#41'><b>Kernel PCA</b></a>
# - <a href = "#16"><b>Data Augmentation</b></a>
# - <a href='#6'><b>Build the Light GBM Model</b></a></pre>

# %% {"_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19", "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5"}
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold,KFold
import warnings
from six.moves import urllib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
# %matplotlib inline
plt.style.use('seaborn')
from scipy.stats import norm, skew

# %% [markdown] {"_uuid": "d150ae0e24acf7d0107ec64ccea13d9745ce45fc"}
# <a id=1><pre><b>Import the Data</b></pre></a>

# %% {"_cell_guid": "79c7e3d0-c299-4dcb-8224-4455121ee9b0", "_uuid": "d629ff2d2480ee46fbb7e2d37f6b5fab8052498a"}
#Load the Data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
features = [c for c in train.columns if c not in ['ID_code', 'target']]

# %% [markdown] {"_uuid": "e711ea5576a8672fce378ede726be247aa789ef1"}
# <a id=11><pre><b>Data Exploration</b></pre></a>

# %% {"_uuid": "0ad0660223a680a8cc777c7526258759fface7a6"}
train.describe()

# %% {"_uuid": "217907a226a7e9425b4445805cde80c5de4feaca"}
train.info()

# %% {"_uuid": "90ca407e625a961a635fde6a21c9f524f024d654"}
train.shape

# %% {"_uuid": "089309dd0b32db21b44152f4bb15b2c7765dfd87"}
train.head(5)

# %% [markdown] {"_uuid": "3548150c4ae4ccd847d84baea5cba641f4fdc0bb"}
# <a id=2><b><pre>Check for the Missing Values.</pre></b></a> 

# %% {"_uuid": "906ec8c811e2d415d47c7f67d8ac23bed0d8699b"}
#Check for Missing Values after Concatination

obs = train.isnull().sum().sort_values(ascending = False)
percent = round(train.isnull().sum().sort_values(ascending = False)/len(train)*100, 2)
pd.concat([obs, percent], axis = 1,keys= ['Number of Observations', 'Percent'])

# %% [markdown] {"_uuid": "bfe81109ea380b1210a3a6d50547058a4ee0e9b5"}
# <pre>There are no missing values in the dataset</pre>

# %% [markdown] {"_uuid": "8d28011134ff59dc25080e743e028bb487b8c366"}
# <pre><a id = 3><b>Visualizing the Satendar Customer Transactions Data</b></a></pre>

# %% [markdown] {"_uuid": "6abbb24cafc26afb4c6f8c52ab6b0353e2698f2e"}
# <pre><a id = 31 ><b>Check for Class Imbalance</b></a></pre>

# %% {"_uuid": "ada8973ebb427bbf9934a911095c1338b9036b35"}
target = train['target']
train = train.drop(["ID_code", "target"], axis=1)
sns.set_style('whitegrid')
sns.countplot(target)

# %% [markdown] {"_uuid": "9bcb709f47ab634bd7ebaa7a9f0574d571e2b30e"}
# <pre><a id = 32 ><b>Distribution of Mean and Standard Deviation</b></a></pre>
#
# <pre>EDA Reference : https://www.kaggle.com/gpreda/santander-eda-and-prediction</pre>

# %% {"_uuid": "60077579a9b2e2b92119d2cebbf29c301c3ee279"}
plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per row in the train and test set")
sns.distplot(train[features].mean(axis=1),color="black", kde=True,bins=120, label='train')
sns.distplot(test[features].mean(axis=1),color="red", kde=True,bins=120, label='test')
plt.legend()
plt.show()

# %% [markdown] {"_uuid": "c5f90ed3f3e3a6c21fd21e7891dd131a981e1f24"}
# <pre>Let's check the distribution of the mean of values per columns in the train and test datasets.</pre>

# %% {"_uuid": "4589fe2bb6b38c8f490057b6c2734aa1c8cf57a5"}
plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per column in the train and test set")
sns.distplot(train[features].mean(axis=0),color="black", kde=True,bins=120, label='train')
sns.distplot(test[features].mean(axis=0),color="red", kde=True,bins=120, label='test')
plt.legend();plt.show()

# %% [markdown] {"_uuid": "17a1f1bd380a50f59f2293071f1fd1cb85d4cace"}
# <pre>Distribution for Standard Deviation</pre>

# %% {"_uuid": "1119bbd9854b60c53eff0f5c024df241cf99a4ff"}
plt.figure(figsize=(16,6))
plt.title("Distribution of std values per rows in the train and test set")
sns.distplot(train[features].std(axis=1),color="blue",kde=True,bins=120, label='train')
sns.distplot(test[features].std(axis=1),color="green", kde=True,bins=120, label='test')
plt.legend(); plt.show()

# %% [markdown] {"_uuid": "2e23ffd37c255be7b01aab8ef6b25d0bd4d2563f"}
# <pre>Let's check the distribution of the standard deviation of values per columns in the train and test datasets.</pre>

# %% {"_uuid": "734b96fd6a8aba302513797962498c906e299653"}
plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per column in the train and test set")
sns.distplot(train[features].mean(axis=0),color="blue", kde=True,bins=120, label='train')
sns.distplot(test[features].mean(axis=0),color="green", kde=True,bins=120, label='test')
plt.legend();plt.show()

# %% [markdown] {"_uuid": "1200ca154b1928043b67fb114d7d0eb93bfbd7e7"}
# <pre>Let's check now the distribution of the mean value per row in the train dataset, grouped by value of target</pre>

# %% {"_uuid": "802622e99a858e7e1be8a56a0dcb32c217769736"}
t0 = train.loc[target == 0]
t1 = train.loc[target == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per row in the train set")
sns.distplot(t0[features].mean(axis=1),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].mean(axis=1),color="green", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()

# %% [markdown] {"_uuid": "bae148d9255104a14c07b0075dbe67084039ada9"}
# <pre>Let's check now the distribution of the mean values per columns in the train and test datasets.</pre>

# %% {"_uuid": "5778c9b5a5b82264a02907471c98aba55e753cf9"}
t0 = train.loc[target == 0]
t1 = train.loc[target == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per column in the train set")
sns.distplot(t0[features].mean(axis=0),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].mean(axis=0),color="green", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()

# %% [markdown] {"_uuid": "dfe2e017dbe64a93c707785b77a2f018c55d2a92"}
# <pre>Let's check now the distribution of the standard deviation  per row in the train dataset, grouped by value of target</pre>

# %% {"_uuid": "03d83a9f09460a7e0e64de7cff618fb903511eb5"}
t0 = train.loc[target == 0]
t1 = train.loc[target == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of standard deviation values per row in the train set")
sns.distplot(t0[features].std(axis=1),color="blue", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].std(axis=1),color="red", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()

# %% [markdown] {"_uuid": "0796b8aa04186d551ae3d92d28e18a548dc09e51"}
# <pre>Let's check now the distribution of standard deviation per columns in the train and test datasets.</pre>

# %% {"_uuid": "8fe584abb584e77e654eb6c768b42eeafda6b784"}
t0 = train.loc[target  == 0]
t1 = train.loc[target  == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of standard deviation values per column in the train set")
sns.distplot(t0[features].std(axis=0),color="blue", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].std(axis=0),color="red", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()

# %% [markdown] {"_uuid": "61fb22a8fdac069232e1584d97e02ca6348c7eea"}
# <pre><a id = 33 ><b>Distribution of Skewness</b></a></pre>
#
# <pre>Let's see now the distribution of skewness on rows in train separated for values of target 0 and 1. We found the distribution is left skewed</pre>

# %% {"_uuid": "a353fcf6b2ce7db7d6c693a2761bc8ac0e005309"}
t0 = train.loc[target == 0]
t1 = train.loc[target == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of skew values per row in the train set")
sns.distplot(t0[features].skew(axis=1),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].skew(axis=1),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()

# %% [markdown] {"_uuid": "3a0d204c325a9b78ff5b242e3b23043645040499"}
# <pre>Let's see now the distribution of skewness on columns in train separated for values of target 0 and 1.</pre>

# %% {"_uuid": "e47c1c00db66e3f43c65efad776bd2bcbea8117d"}
t0 = train.loc[target == 0]
t1 = train.loc[target == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of skew values per column in the train set")
sns.distplot(t0[features].skew(axis=0),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].skew(axis=0),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()

# %% [markdown] {"_uuid": "52dc95b188e82d5e55503348b8db57abfb385ca2"}
# <pre><a id = 34 ><b>Distribution of Kurtosis</b></a></pre>

# %% [markdown] {"_uuid": "b3d635fc2ccd5d0ad662413ccff46e062a01a13c"}
# <pre>Let's see now the distribution of kurtosis on rows in train separated for values of target 0 and 1. We found the distribution to be Leptokurtic</pre>

# %% {"_uuid": "a0785f3344f18166d838b50ecfb05901ad2180c8"}
t0 = train.loc[target == 0]
t1 = train.loc[target == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of kurtosis values per row in the train set")
sns.distplot(t0[features].kurtosis(axis=1),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].kurtosis(axis=1),color="green", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()

# %% [markdown] {"_kg_hide-input": true, "_kg_hide-output": true, "_uuid": "736f0bde864b3bf327be491a0d820593415aa3f5"}
# <pre>Let's see now the distribution of kurtosis on columns in train separated for values of target 0 and 1.</pre>

# %% {"_uuid": "8b72cdd5a6f9b1db419fdd35e44974e219a9d376"}
t0 = train.loc[target == 0]
t1 = train.loc[target == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of kurtosis values per column in the train set")
sns.distplot(t0[features].kurtosis(axis=0),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].kurtosis(axis=0),color="green", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()

# %% [markdown] {"_uuid": "374e9be094d1adaf17888cb16aea2f10093edd9e"}
# <a id=4><pre><b>Principal Component Analysis to check Dimentionality Reduction<b></pre></a>

# %% {"_uuid": "0af73d37cc75d3685fcb5f8c2702ad8758070b94"}
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)         
PCA_train_x = PCA(2).fit_transform(train_scaled)
plt.scatter(PCA_train_x[:, 0], PCA_train_x[:, 1], c=target, cmap="copper_r")
plt.axis('off')
plt.colorbar()
plt.show()

# %% [markdown] {"_uuid": "2482fcb3497bcc3b7fe7f27256e408ff98324de2"}
# <pre><a id = 41><b>Kernel PCA (Since the Graph above doesn't represent meaningful analysis)</b></a></pre>

# %% {"_uuid": "9206e909ab4be625c94811af6bd0b676f626de22"}
from sklearn.decomposition import KernelPCA

lin_pca = KernelPCA(n_components = 2, kernel="linear", fit_inverse_transform=True)
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
sig_pca = KernelPCA(n_components = 2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)


plt.figure(figsize=(11, 4))
for subplot, pca, title in ((131, lin_pca, "Linear kernel"), (132, rbf_pca, "RBF kernel, $\gamma=0.04$"), 
                            (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):
       
    PCA_train_x = PCA(2).fit_transform(train_scaled)
    plt.subplot(subplot)
    plt.title(title, fontsize=14)
    plt.scatter(PCA_train_x[:, 0], PCA_train_x[:, 1], c=target, cmap="nipy_spectral_r")
    plt.xlabel("$z_1$", fontsize=18)
    if subplot == 131:
        plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)

plt.show()


# %% [markdown] {"_uuid": "5b7a96339294daeedba94abaee4fbe6f16e69f2e"}
# <pre>Since PCA hasn't been useful, I decided to proceed with the existing dataset</pre>

# %% [markdown] {"_uuid": "96861473dd6cb2de3377a47684ece1714e1ab072"}
# <pre><a id = 16><b>Data Augmentation</b></a></pre>

# %% {"_uuid": "dfd26c446ff80f323791fbdbbbf158d355ee7267"}
def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y


# %% [markdown] {"_uuid": "a37f046be743d0086a2fc6094d78d7b9cab78055"}
# <pre><a id = 6><b>Build the Light GBM Model</b></a></pre>

# %% {"_uuid": "d418b9c44ef2f96b02db44d70aacbca61fe0952f"}
param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.335,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.041,
    'learning_rate': 0.0083,
    'max_depth': -1,
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': -1
}

# %% {"_uuid": "fc22f099688ce4928a44f1c68cd16d6b8473e207"}
train.shape

# %% {"_uuid": "8b4f1d5f4aef4730673a8a6bbb2e828c2f92e2a5"}
num_folds = 11
features = [c for c in train.columns if c not in ['ID_code', 'target']]

folds = KFold(n_splits=num_folds, random_state=2319)
oof = np.zeros(len(train))
getVal = np.zeros(len(train))
predictions = np.zeros(len(target))
feature_importance_df = pd.DataFrame()

print('Light GBM Model')
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    
    X_train, y_train = train.iloc[trn_idx][features], target.iloc[trn_idx]
    X_valid, y_valid = train.iloc[val_idx][features], target.iloc[val_idx]
    
    X_tr, y_tr = augment(X_train.values, y_train.values)
    X_tr = pd.DataFrame(X_tr)
    
    print("Fold idx:{}".format(fold_ + 1))
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_valid, label=y_valid)
    
    clf = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=5000, early_stopping_rounds = 4000)
    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    getVal[val_idx]+= clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration) / folds.n_splits
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

# %% {"_uuid": "f9dc76139cb15edf957be0a8400e6de33c14e655"}
cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)
best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,26))
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('LightGBM Features (averaged over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')

# %% {"_uuid": "137cf3c3924422e1a15ac63f4e259b86db86c2c5"}
num_sub = 26
print('Saving the Submission File')
sub = pd.DataFrame({"ID_code": test.ID_code.values})
sub["target"] = predictions
sub.to_csv('submission{}.csv'.format(num_sub), index=False)
getValue = pd.DataFrame(getVal)
getValue.to_csv("Validation_kfold.csv")
