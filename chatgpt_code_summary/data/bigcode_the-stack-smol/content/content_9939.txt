import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import time

#Criteo's CTR Prediction Challenge

#Creating a list of the numerical and categorical variables
intnames = []
catnames = []
for i in range(13):
    intnames += ['i'+ str(i+1)]
for i in range(26):
    catnames += ['c'+ str(i+1)]
colnames = ['clicked'] + intnames + catnames

#Load Data (500,000 rows) and name columns
ds = pd.read_csv("train.txt", nrows=500000, sep='\t', header=None, names = colnames)

#Basic info of dataset
ds.info()
ds['clicked'].mean()

#Number of categories per each category variable
categoriesPerVariable = {}
for var in catnames:
    varList = ds[var].tolist()
    varUnique = set(varList)
    print(var, len(varUnique))
    categoriesPerVariable[var] = len(varUnique)

catnamesFinal = []
#Delete variables with more than 100 categories
for var in categoriesPerVariable:
    if categoriesPerVariable[var] > 100:
        ds = ds.drop(var, 1)
        print(var, 'DELETED')
    else: catnamesFinal += [var]

ds.info()

#Create dummy variables:
for var in catnamesFinal:
    ds = pd.concat([ds, pd.get_dummies(ds[var], prefix = var, prefix_sep = '_')], axis=1)
    ds = ds.drop(var, axis=1)
    print('Created dummy variables for: ', var)
    
ds.shape

#Creating train and test datasets
y = ds.clicked
x_cols = set(ds.columns)
x_cols.remove('clicked')
X = ds[list(x_cols)]

#Train, test and Validation Sets (60%, 20%, 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=0.5)

#More Preprocessing
#   - Fill NaN values in X_train, X_test, X_val with the mean of X_train
X_train[intnames] = X_train[intnames].fillna(X_train[intnames].mean())
X_test[intnames] = X_test[intnames].fillna(X_train[intnames].mean())
X_val[intnames] = X_val[intnames].fillna(X_train[intnames].mean())

#Dataset with PCA
#Choosing the number of components
from sklearn.decomposition import PCA
for e in range(10):
    pca1 = PCA(n_components=e)
    pca1.fit(X_train)
    exp_var = 0
    for i in pca1.explained_variance_ratio_:
        exp_var += i
    print(e, round(exp_var,3))

pca = PCA(n_components=5)
pca.fit(X_train)
X_train_PCA = pd.DataFrame(pca.transform(X_train))
X_test_PCA = pd.DataFrame(pca.transform(X_test))
X_val_PCA = pd.DataFrame(pca.transform(X_val))

'''
###########################################
# FUNCTIONS
###########################################
''' 

def ROCcurve(y_pred, y_test):
# Compute ROC curve and ROC area for each class
    n_classes = y_pred.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    y_test1 = []
    for index, row in y_test.iteritems():
        if row == 0: y_test1 += [[1,0]]
        else: y_test1 += [[0,1]]
    y_test1 = np.array(y_test1)
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test1[:,i], y_pred[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test1.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr, tpr, roc_auc

#Precision Recall Curve
def precision_recall(y_test, y_pred):
    ydp = []
    for i in range(len(y_pred)): ydp+= [y_pred[i][1]]
    precision, recall, _ = precision_recall_curve(y_test, ydp)
    return precision, recall

def ypd(y_pred):
    ydp = []
    for i in range(len(y_pred)): ydp+= [y_pred[i][1]]
    return ydp    

#Return all the Algorithm Curves Info
def algorithmCurvesInfo(alg_name, y_pred, y_test):
    algDict = {}
    algDict['alg_name'] = alg_name
    algDict['fpr'], algDict['tpr'], \
        algDict['roc_auc'] = ROCcurve(y_pred, y_test)
    algDict['precision'], algDict['recall'], = precision_recall(y_test, y_pred)
    algDict['average_precision'] = average_precision_score(y_test, ypd(y_pred))
    return algDict

#PLOT ROC CURVE
def plotROC(alg_fpr_tpr_rocDict, color_paletteList, tuple_size, path_name):
    colors = []
    for p in color_paletteList:
        for c in plt.get_cmap(p).colors:
            colors += [c]
    #Dict with key --> name of algorithm (dict)
    #Each algorithm dict:
    #   - fpr
    #   - tpr
    #   - roc_auc
    #   - alg_name: algorithm name to be shown
    plt.figure(figsize=tuple_size)
    col = 0
    for al in alg_fpr_tpr_rocDict:
        fpr = alg_fpr_tpr_rocDict[al]['fpr']
        tpr = alg_fpr_tpr_rocDict[al]['tpr']
        roc_auc = alg_fpr_tpr_rocDict[al]['roc_auc']
        alg_name = alg_fpr_tpr_rocDict[al]['alg_name']
        plt.plot(fpr[1], tpr[1], color= colors[col], alpha = 0.7, 
                 label= alg_name + ' (area = %0.3f)' % roc_auc[1])
        col += 1
        
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])        
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves per each algorithm')
    plt.legend(loc="lower right")
    plt.savefig(path_name + '_ROC-AUC.png')
    plt.show()

#Plot the Precision-Recall Curve
def plotPrecisionRecall(alg_pre_recDict, color_paletteList, tuple_size, path_name):
    colors = []
    for p in color_paletteList:
        for c in plt.get_cmap(p).colors:
            colors += [c]
    col = 0
    plt.figure(figsize=tuple_size)
    for al in alg_pre_recDict:
        recall = alg_pre_recDict[al]['recall']
        precision = alg_pre_recDict[al]['precision']
        average_precision = alg_pre_recDict[al]['average_precision']
        alg_name = alg_pre_recDict[al]['alg_name']
        '''
        plt.step(recall, precision, color=colors[col], alpha=0.8, where='post', \
                 label= alg_name + ' (area = %0.3f)'.format(average_precision))
        '''
        plt.plot(recall, precision, color=colors[col], alpha=0.8, \
                 label= alg_name + ' (area = %0.3f)' % average_precision)
        col += 1

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="upper right")
    plt.title('Precision-Recall curve for CLICKED')
    plt.savefig(path_name + '_PrecisionRecall.png')
    plt.show()

#Algorithm Process Automation
def algorithmAutomat(algorithm, X_train, y_train, X_test, name):
    algDict = {}
    train_s = time.time()
    algorithm.fit(X_train, y_train)
    train_e = time.time()
    pred_s = time.time()
    y_pred = algorithm.predict_proba(X_test)
    pred_e = time.time()
    algDict = algorithmCurvesInfo(name, y_pred, y_test)
    algDict['train_time'] = round(train_e - train_s,2)
    algDict['predict_time'] = round(pred_e - pred_s,2)
    algDict['model'] = algorithm
    print(name + ' Prediction calculated')
    print('Elapsed time: ' + str(round(pred_e-train_s,2)) + ' seconds')
    return algDict

#Algorithm Validation Prediction and Curves
def algorithmValidation(model, X_validation, y_validation, name):
    algDict = {}
    start = time.time()
    y_pred = model.predict_proba(X_validation)
    end = time.time()
    algDict = algorithmCurvesInfo(name, y_pred, y_validation)
    algDict['prediction_time'] = end - start
    print(name + ' Prediction calculated')
    return algDict


'''
###########################################
# ALGORITHMS
###########################################
''' 
#Path where I will save the ROC and Precision-Recall curves
path = os.getcwd() + '/graphs/'

#Dictionaries to save algorithms' results for dataset with and without PCA
algs = {}
algsPCA = {}

######################################
# Logistic Regression
######################################
from sklearn.linear_model import LogisticRegression

#Parameter tuning options
regularizers = ['l1', 'l2']
C = [0.001,0.01,0.1,1,10,100,1000]

algs['lr'] = {}
for r in regularizers:
    for c in C:
        #Algorithm name based on tuning options
        name = 'LogReg_' + str(r) + '_' + str(c)
        logreg = LogisticRegression(penalty = r, C = c ,random_state = 0)
        algs['lr'][name] = algorithmAutomat(logreg, X_train, y_train, X_test, name)

algsPCA['lr'] = {}
for r in regularizers:
    for c in C:
        name = 'LogRegPCA_' + str(r) + '_' + str(c)
        logreg = LogisticRegression(penalty = r, C = c ,random_state = 0)
        algsPCA['lr'][name] = algorithmAutomat(logreg, X_train_PCA, y_train, X_test_PCA, name)

#Plots
plotROC(algs['lr'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'LR')
plotROC(algsPCA['lr'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'LRpca')
plotPrecisionRecall(algs['lr'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'LR')
plotPrecisionRecall(algsPCA['lr'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'LRpca')

######################################
# Random Forest
######################################
from sklearn.ensemble import RandomForestClassifier

n_estim = [2, 10, 50, 100, 1000]
max_d = [None, 2, 5, 10, 50]

algsPCA['rf'] = {}
for n in n_estim:
    for m in max_d:
        name = 'RandForPCA_est' + str(n) + '_depth' + str(m)
        rf = RandomForestClassifier(n_estimators = n, max_depth=m, random_state=0)
        algsPCA['lr'][name] = algorithmAutomat(rf, X_train_PCA, y_train, X_test_PCA, name)

algs['rf'] = {}
for n in n_estim:
    for m in max_d:
        name = 'RandFor_est' + str(n) + '_depth' + str(m)       
        rf = RandomForestClassifier(n_estimators = n, max_depth=m, random_state=0)
        algs['rf'][name] = algorithmAutomat(rf, X_train, y_train, X_test, name)

plotROC(algs['rf'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'RF')
plotROC(algsPCA['rf'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'RFpca')
plotPrecisionRecall(algs['rf'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'RF')
plotPrecisionRecall(algsPCA['rf'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'RFpca')

######################################
# K-nearest neighbors
######################################
from sklearn.neighbors import KNeighborsClassifier

algsPCA['knn'] = {}

for k in [5, 10, 20, 50, 100, 200]:
    name = 'KNN_PCA_' + str(k)
    knn = KNeighborsClassifier(n_neighbors=k)
    algsPCA['knn'][name] = algorithmAutomat(knn, X_train_PCA, y_train, X_test_PCA, name)
    
algs['knn'] = {}

for k in [5, 10, 20, 50, 100, 200]:
    name = 'KNN_' + str(k)
    knn = KNeighborsClassifier(n_neighbors=k)
    algs['knn'][name] = algorithmAutomat(knn, X_train, y_train, X_test, name)

plotROC(algs['knn'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'KNN')
plotROC(algs['knn']['knn'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'KNNpca')
plotPrecisionRecall(algs['knn'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'KNN')
plotPrecisionRecall(algs['knn']['knn'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'KNNpca')

######################################
# Naive Bayes
######################################
from sklearn.naive_bayes import GaussianNB

algsPCA['nbayes'] = {}
algs['nbayes'] = {}

gnb = GaussianNB()
name = 'NaiveBayes_PCA'
algsPCA['nbayes'][name] = algorithmAutomat(gnb, X_train_PCA, y_train, X_test_PCA, name)

algs['nbayes'] = {}
gnb = GaussianNB()
name = 'NaiveBayes'
algs['nbayes'][name] = algorithmAutomat(gnb, X_train, y_train, X_test, name)

plotROC(algs['nbayes'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'NB')
plotPrecisionRecall(algs['nbayes'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'NB')
plotROC(algsPCA['nbayes'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'NB')
plotPrecisionRecall(algsPCA['nbayes'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'NBpca')

######################################
# AdaBoost
######################################
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

n_estim = [2, 10, 50]
max_d = [2, 10, 50, None]

algsPCA['adab'] = {}
for n in n_estim:
    for m in max_d:
        name = 'AdaBoost_PCA_est' + str(n) + '_depth' + str(m)
        bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=m),
            algorithm="SAMME", n_estimators=n)
        algsPCA['adab'][name] = algorithmAutomat(bdt, X_train_PCA, y_train, X_test_PCA, name)
        
algs['adab'] = {}
for n in n_estim:
    for m in max_d:
        name = 'AdaBoost_est' + str(n) + '_depth' + str(m)
        bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=m),
            algorithm="SAMME", n_estimators=n)
        algs['adab'][name] = algorithmAutomat(bdt, X_train, y_train, X_test, name)    

plotROC(algs['adab'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'AB')
plotROC(algsPCA['adab'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'ABpca')
plotPrecisionRecall(algs['adab'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'AB')
plotPrecisionRecall(algsPCA['adab'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'ABpca')

######################################
# Linear Discriminant Analysis
######################################
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

algsPCA['lda'] = {}
lda = LDA()
name = 'LDA_PCA'
algsPCA['lda'] [name] = algorithmAutomat(lda, X_train_PCA, y_train, X_test_PCA, name)

algs['lda'] = {}
lda = LDA()
name = 'LDA'
algs['lda'][name] = algorithmAutomat(lda, X_train, y_train, X_test, name)

plotROC(algs['lda'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'LDA')
plotPrecisionRecall(algs['lda'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'LDA')
plotROC(algsPCA['lda'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'LDA')
plotPrecisionRecall(algsPCA['lda'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'LDApca')

######################################
# Gradient Boosting
######################################
from sklearn.ensemble import GradientBoostingClassifier

learning_rate = [0.1, 1]
max_depth = [3,5]
loss = ['deviance', 'exponential']

algsPCA['gradbo'] = {}
for l in learning_rate:
    for m in max_depth:
        for lo in loss:
            name = 'GradBoost_PCA_lr' + str(l) + '_depth' + str(m) + '_loss-' + lo
            gbc = GradientBoostingClassifier(learning_rate = l, max_depth = m, loss = lo)
            algsPCA['gradbo'][name] = algorithmAutomat(gbc, X_train_PCA, y_train, X_test_PCA, name)

algs['gradbo'] = {}
for l in learning_rate:
    for m in max_depth:
        for lo in loss:
            name = 'GradBoost_lr' + str(l) + '_depth' + str(m) + '_loss-' + lo
            gbc = GradientBoostingClassifier(learning_rate = l, max_depth = m, loss = lo)
            algs['gradbo'][name] = algorithmAutomat(gbc, X_train, y_train, X_test, name)    

plotROC(algs['gradbo'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'GB')
plotROC(algsPCA['gradbo'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'GBpca')
plotPrecisionRecall(algs['gradbo'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'GB')
plotPrecisionRecall(algsPCA['gradbo'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'GBpca')


######################################
# NEURAL NETWORKS: MLP
######################################
#File mlp_train.py  included in this repository
#This file contains the needed functions
import mlp_train as mlp

from keras.utils import to_categorical
#Creating dummy variables for the response variable (needed in Neural Nets)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
algs['mlp'] = {}
#Load model trained in MLP.py

#Parameters:
baseName = 'MLP_'
batch_size = 200
epochs = 20
optimizer = ['adam', 'rmsprop']
denseLayers = [2, 3]
layerNeurons = [200, 500]
dropout = [True, False]
dropoutRate = 0.3
n_classes = 2

for o in optimizer:
    for d in denseLayers:
        for do in dropout:
            for ln in layerNeurons:
                name, y_pred_mlp, train_time = mlp.NeuralNetProcess(baseName, o, batch_size, epochs, d, ln,
                                    do, dropoutRate, X_train, X_test, y_train_cat, y_test_cat, n_classes)
                algs['mlp'][name] = algorithmCurvesInfo(name, y_pred_mlp, y_test)
                algs['mlp'][name]['train_time'] = train_time
                algs['mlp'][name]['predict_time'] = None
                
plotROC(algs['mlp'], ['Set1', 'Set2', 'Set3', 'Set1'], (10,8), path + 'MLP')
plotPrecisionRecall(algs['mlp'], ['Set1', 'Set2', 'Set3', 'Set1'], (10,8), path + 'MLP')


'''
###########################################
# TEST EXPORTING SUMMARIES
###########################################
''' 
#Exporting summary info to csv file
headers = ['HAS_PCA', 'ALGORITHM', 'ALG_NAME', 'TRAIN_TIME', 'PREDICT_TIME', 'AVERAGE_PRECISION', 'ROC_AUC']
rows = []
for a in algsPCA:
    print('---------------',a)
    for k in algsPCA[a]:
        row = []
        row += ['PCA']
        row += [a]
        row += [k]
        row += [str(algsPCA[a][k]['train_time'])]
        row += [str(algsPCA[a][k]['predict_time'])]
        row += [str(algsPCA[a][k]['average_precision'])]
        row += [str(algsPCA[a][k]['roc_auc'][1])]
        rows += [row]

for a in algs:
    print('---------------',a)
    for k in algs[a]:
        row = []
        row += ['REG']
        row += [a]
        row += [k]
        row += [str(algs[a][k]['train_time'])]
        row += [str(algs[a][k]['predict_time'])]
        row += [str(algs[a][k]['average_precision'])]
        row += [str(algs[a][k]['roc_auc'][1])]
        rows += [row]            

csvfile = ', '.join(headers)
for r in rows:
    csvfile += '\n' + ', '.join(r)

f = open(os.getcwd() + "\\algorithmsDataset.csv",'w')
f.write(csvfile)
f.close()

'''
###########################################
# VALIDATION MODELS
###########################################
''' 

#Select best tuned model for each algorithm and store the list
#Established a limit_train_time and limit_predict_time.
bestAlgs = {}
limitTrainTime = 400
limitPredictTime = 200
bestAlgs['PCA'] = {}
for a in algsPCA:
    balg = ''
    roc = 0
    for k in algsPCA[a]:
        if algsPCA[a][k]['roc_auc'][1] > roc and algsPCA[a][k]['train_time'] < limitTrainTime and algsPCA[a][k]['predict_time'] < limitPredictTime:
            roc = algsPCA[a][k]['roc_auc'][1]
            balg = k
    bestAlgs['PCA'][balg] = roc

bestAlgs['REG'] = {}
for a in algs:
    balg = ''
    roc = 0
    for k in algs[a]:
        if algs[a][k]['roc_auc'][1] > roc and algs[a][k]['train_time'] < limitTime and algs[a][k]['predict_time'] < limitPredictTime:
            roc = algs[a][k]['roc_auc'][1]
            balg = k
    bestAlgs['REG'][balg] = roc


'''
###########################################
# VALIDATION PREDICTIONS
###########################################
''' 
#Predict results using the validation set for each selected model

VALalgs = {}
VALalgs['PCA'] = {}
for k in bestAlgs['PCA']: print(k)

name = 'LogRegPCA_l1_100'
VALalgs['PCA'][name] = algorithmValidation(algsPCA['lr'][name]['model'], X_val_PCA, y_val, name)

name = 'RandForPCA_est100_depth10'
VALalgs['PCA'][name] = algorithmValidation(algsPCA['rf'][name]['model'], X_val_PCA, y_val, name)

name = 'KNN_PCA_100'
VALalgs['PCA'][name] = algorithmValidation(algsPCA['knn'][name]['model'], X_val_PCA, y_val, name)

name = 'NaiveBayes_PCA'
VALalgs['PCA'][name] = algorithmValidation(algsPCA['nbayes'][name]['model'], X_val_PCA, y_val, name)

name = 'AdaBoost_PCA_est50_depth10'
VALalgs['PCA'][name] = algorithmValidation(algsPCA['adab'][name]['model'], X_val_PCA, y_val,name)

name = 'GradBoost_PCA_lr0.1_depth5_loss-deviance'
VALalgs['PCA'][name] = algorithmValidation(algsPCA['gradbo'][name]['model'], X_val_PCA, y_val, name)

name = 'LDA_PCA'
VALalgs['PCA'][name] = algorithmValidation(algsPCA['lda'][name]['model'], X_val_PCA, y_val, name)

VALalgs['REG'] = {}
for k in bestAlgs['REG']: print(k)

name = 'LogReg_l1_0.1'
VALalgs['REG'][name] = algorithmValidation(algs['lr'][name]['model'], X_val, y_val, name)

name = 'RandFor_est100_depth50'
VALalgs['REG'][name] = algorithmValidation(algs['rf'][name]['model'], X_val, y_val, name)

name = 'KNN_100'
VALalgs['REG'][name] = algorithmValidation(algs['knn'][name]['model'], X_val, y_val, name)

name = 'NaiveBayes'
VALalgs['REG'][name] = algorithmValidation(algs['nbayes'][name]['model'], X_val, y_val, name)

name = 'AdaBoost_est50_depth10'
VALalgs['REG'][name] = algorithmValidation(algs['adab'][name]['model'], X_val, y_val, name)

name = 'GradBoost_lr0.1_depth5_loss-deviance'
VALalgs['REG'][name] = algorithmValidation(algs['gradbo'][name]['model'], X_val, y_val, name)

name = 'LDA'
VALalgs['REG'][name] = algorithmValidation(algs['lda'][name]['model'], X_val, y_val, name)

name = 'MLP_rmsprop_b200_e20_DL2_200_drop-False_0.3'
bestModelPath = os.getcwd() + '/NNbestModel/'
bestModelPathLoss = bestModelPath + 'model_loss_' + name + '.hdf5'
y_pred_mlp, prediction_time = mlp.NeuralNetPredict(bestModelPathLoss, X_val)
VALalgs['REG'][name] = algorithmCurvesInfo(name, y_pred_mlp, y_val)
VALalgs['REG'][name]['prediction_time'] = prediction_time

#Plot & Save
plotROC(VALalgs['PCA'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'Val_PCA')
plotROC(VALalgs['REG'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'Val_REG')
plotPrecisionRecall(VALalgs['PCA'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'Val_PCA')
plotPrecisionRecall(VALalgs['REG'], ['Set1', 'Set2', 'Set3'], (10,8), path + 'Val_REG')

'''
###########################################
# VALIDATION EXPORTING SUMMARIES
###########################################
''' 

#Exporting validation set data to csv
headers = ['HAS_PCA', 'ALGORITHM', 'ALG_NAME', 'TRAIN_TIME', 'PREDICT_TIME', 'AVERAGE_PRECISION', 'ROC_AUC']
val_rows = []
for a in algsPCA:
    for k in algsPCA[a]:
        if k in VALalgs['PCA']:
            print('---------------',a)
            row = []
            row += ['PCA']
            row += [a]
            row += [k]
            row += [str(algsPCA[a][k]['train_time'])]
            row += [str(VALalgs['PCA'][k]['prediction_time'])]
            row += [str(VALalgs['PCA'][k]['average_precision'])]
            row += [str(VALalgs['PCA'][k]['roc_auc'][1])]
            val_rows += [row]

for a in algs:
    for k in algs[a]:
        if k in VALalgs['REG']:
            print('---------------',a)
            row = []
            row += ['REG']
            row += [a]
            row += [k]
            row += [str(algs[a][k]['train_time'])]
            row += [str(VALalgs['REG'][k]['prediction_time'])]
            row += [str(VALalgs['REG'][k]['average_precision'])]
            row += [str(VALalgs['REG'][k]['roc_auc'][1])]
            val_rows += [row]      

csvfile = ', '.join(headers)
for r in val_rows:
    csvfile += '\n' + ', '.join(r)

f = open(os.getcwd() + "\\algorithmsValidationDataset.csv",'w')
f.write(csvfile)
f.close()
