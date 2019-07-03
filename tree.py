# -*- coding: utf-8 -*-
from inspect import signature
from xgboost.sklearn import XGBClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
from pandas import DataFrame
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import svm
from sklearn.metrics import precision_recall_curve
import math
import graphviz
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

def pre_advanced():
    dataset=pd.read_csv("/Users/shangliufangjian/Documents/工作/蜗牛保险/datas.csv",encoding='gb2312')

    dataset_y=[]
    size=len(dataset)
    kidnum=np.zeros((size,1))
    #需要计数的变量们
    haspartner=np.zeros((size,1))
    parentsnum=np.zeros((size,1))
    hasillness=np.ones((size,1))
    hashouseloan=np.ones((size,1))
    hascarloan=np.ones((size,1))

    datasettobedummied=dataset.drop(['生日','智诊时间','子女'],axis=1)
    #dataset['年龄']
    datasettobedummied=datasettobedummied[['性别(1：男 2：女)','工作','主要出行工具是否为私家车(1：是 2：否)','家庭成员是否经常出差(1：是 2：否)','家庭成员是否有社保(1有 2无)']].fillna('未知')

    dummies=pd.get_dummies(datasettobedummied[['性别(1：男 2：女)','工作','主要出行工具是否为私家车(1：是 2：否)','家庭成员是否经常出差(1：是 2：否)','家庭成员是否有社保(1有 2无)']])
#计数变量 有无角色 有无房车贷
    for i in range(size):
        if(dataset.iloc[i,1]=='父母'):
            parentsnum[i]=1
        elif (dataset.iloc[i,1]=='子女'):
            kidnum[i]=1
  #          dataset.iloc[i, 1] =dataset.iloc[i,2]
        elif (dataset.iloc[i,1]=='配偶'):
            haspartner[i]=1
        if (dataset.iloc[i, 14] == '无疾病'or pd.isna(dataset.iloc[i, 14])):
            hasillness[i] = 0
        if (dataset.iloc[i, 18] == 0 or pd.isna(dataset.iloc[i, 18])):
            hashouseloan[i] = 0
        if (dataset.iloc[i, 22] == 0 or pd.isna(dataset.iloc[i, 22])):
            hascarloan[i] = 0

    dataset['parentsnum']=parentsnum
    dataset['haspartner']=haspartner
    dataset['kidnum']=kidnum
    dataset['hasillness'] = hasillness
    dataset['hashouseloan']=hashouseloan
    dataset['hascarloan']=hascarloan
    #dataset2 = dataset.groupby(dataset['用户id']).sum()
    #characterdummies = dataset2[['parentsnum', 'haspartner', 'kidnum']]

    data_tobediscretize_age=dataset[['年龄']].fillna(-100)
    #data_tobediscretize_salary=dataset[['年收入']].fillna(0)

#对年收入异常值的修正
    dataset['年收入'] = dataset['年收入'].fillna(0)
    dataset.loc[dataset['年收入'] > 1000, '年收入'] = None
    data_income = dataset['年收入'].groupby(dataset['工作']).mean()
    dic = data_income.to_dict()
    d1 = dataset.loc[dataset['年收入'].isna(), ['年收入', '工作']]
    for i in range(len(d1)):
        d1.iloc[i, 0] = dic.get(d1.iloc[i, 1])
    dataset.loc[dataset['年收入'].isna()] = d1

    data_tobediscretize_salary = dataset[['年收入']]
    dis1 = KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile')
    dis2 = KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='uniform')
    data_discretized_age = dis1.fit_transform(data_tobediscretize_age)
    data_discretized_salary = dis2.fit_transform(data_tobediscretize_salary)

    data_discretized_age=pd.DataFrame(data_discretized_age)
    data_discretized_salary=pd.DataFrame(data_discretized_salary)
    data_discretized_age.columns=['年龄1','年龄2','年龄3','年龄4','年龄5']
    data_discretized_salary.columns = ['年收入1', '年收入2', '年收入3', '年收入4', '年收入5'] #收入单位不统一


    #newdataset = pd.concat([dataset[['用户id', '角色','parentsnum','haspartner','kidnum','hasillness']], dummies, data_discretized_age,data_discretized_salary], axis=1)
    newdataset_advanced = pd.concat([dataset[['用户id', '角色', 'hasillness','parentsnum','haspartner','kidnum','hashouseloan','hascarloan']], dummies,data_discretized_age,data_discretized_salary], axis=1)
    #newdataset = pd.concat([newdataset, data_discretized_age,data_discretized_salary],axis=1)

    newdataset_advanced.set_index('用户id', inplace=True)
    newdataset_advanced=newdataset_advanced.sort_index()

    ##上面的代码主要是简化数据，增加有无相关亲人的dummy


    #进阶数据

    newdataset2_advanced=pd.pivot_table(newdataset_advanced,index=['用户id','角色'],aggfunc=[np.sum])


    dataset_x_advanced=newdataset2_advanced.unstack().fillna(0)
    #多层合并

    dataset_x_advanced.columns = ["_".join(x) for x in dataset_x_advanced.columns.ravel()]
    #print(dataset_x_advanced.head(2))
    dataset_x_advanced = dataset_x_advanced.drop(['sum_haspartner_子女','sum_haspartner_父母','sum_kidnum_本人','sum_kidnum_父母',
                                                  'sum_kidnum_配偶','sum_parentsnum_子女','sum_parentsnum_本人','sum_parentsnum_配偶',
                                                  'sum_hashouseloan_子女','sum_hascarloan_子女','sum_hashouseloan_父母','sum_hascarloan_父母'], axis = 1)
    # print(dataset_x_advanced.columns)
    print(dataset_x_advanced.shape)
    dataset_x_advanced = dataset_x_advanced.drop(['sum_工作_室内制造业（装修、流水线工人）_子女','sum_工作_室内制造业（装修、流水线工人）_父母',
                                                  'sum_工作_室内轻体力（行政、管理人员）_子女','sum_工作_室内轻体力（行政、管理人员）_父母',
                                                  'sum_工作_室内重体力（程序员）_子女', 'sum_工作_室内重体力（程序员）_父母',
                                                  'sum_工作_家庭主妇/主男_子女', 'sum_工作_家庭主妇/主男_父母',
                                                  'sum_工作_家庭主妇/夫_子女', 'sum_工作_家庭主妇/夫_父母',
                                                  'sum_工作_家庭主妇/男_子女', 'sum_工作_家庭主妇/男_父母',
                                                  'sum_工作_户外复杂工作（工程师、建筑工人）_子女', 'sum_工作_户外复杂工作（工程师、建筑工人）_父母',
                                                  'sum_工作_户外简单工作（导游、司机）_子女', 'sum_工作_户外简单工作（导游、司机）_父母',
                                                  'sum_主要出行工具是否为私家车(1：是 2：否)_1.0_子女','sum_主要出行工具是否为私家车(1：是 2：否)_2.0_子女',
                                                  'sum_主要出行工具是否为私家车(1：是 2：否)_1.0_父母','sum_主要出行工具是否为私家车(1：是 2：否)_2.0_父母',
                                                  'sum_主要出行工具是否为私家车(1：是 2：否)_未知_本人','sum_主要出行工具是否为私家车(1：是 2：否)_未知_配偶',
                                                  # 'sum_是否赡养父母(1是 2否)_子女','sum_是否赡养父母(1是 2否)_父母',
                                                  'sum_家庭成员是否有社保(1有 2无)_1.0_子女','sum_家庭成员是否有社保(1有 2无)_2.0_子女',
                                                  'sum_家庭成员是否有社保(1有 2无)_1.0_父母','sum_家庭成员是否有社保(1有 2无)_2.0_父母',
                                                  'sum_家庭成员是否有社保(1有 2无)_未知_本人','sum_家庭成员是否有社保(1有 2无)_未知_配偶',
                                                  'sum_家庭成员是否经常出差(1：是 2：否)_1.0_子女','sum_家庭成员是否经常出差(1：是 2：否)_2.0_子女',
                                                  'sum_家庭成员是否经常出差(1：是 2：否)_1.0_父母','sum_家庭成员是否经常出差(1：是 2：否)_2.0_父母',
                                                  'sum_家庭成员是否经常出差(1：是 2：否)_未知_本人','sum_家庭成员是否经常出差(1：是 2：否)_未知_配偶',
                                                   ], axis = 1)
    print(dataset_x_advanced.shape)
    #dataset3=dataset.groupby([dataset['用户id'],dataset['角色']]).sumbxf9()
    for i in dataset['保费']:
        if(np.isnan(i)):
            dataset_y.append(0)
        else:
            dataset_y.append(1)
    dataset['买了']=dataset_y


    dataset_fory=dataset.groupby(dataset['用户id']).mean()
    alldata=pd.concat([dataset_x_advanced,dataset_fory],axis=1,join_axes=[dataset_x_advanced.index])#同样，只保存ID下有本人资料的数据

    dataset_y=alldata['买了']
    X_train, X_test, y_train, y_test = train_test_split(dataset_x_advanced, dataset_y, test_size=0.2, random_state=0)
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = SMOTE(kind='borderline1').fit_sample(X_train, y_train)
    #列名须重命名
    colnames_x=X_train.columns.values

    X_resampled=pd.DataFrame(X_resampled,columns=colnames_x)

    X_resampled.columns=colnames_x

    #y_resampled.columns=colnames_y
    #X_resampled, y_resampled = ros.fit_sample(X_train, y_train)
    #print(X_test.head(2))
    #print(y_resampled.value_counts())

    return X_resampled, X_test, y_resampled, y_test

def mytree(X_train, X_test, y_train, y_test,depth):

    clf = tree.DecisionTreeClassifier(max_depth=depth)
    clf = clf.fit(X_train, y_train)
    pred=clf.predict(X_test)
    score = metrics.f1_score(y_test, pred, average='macro')
    print(score)
    print(classification_report(y_test, pred))
    dot_data = tree.export_graphviz(clf, out_file=None)
    #getpr(y_test, pred)
    getroc(y_test, pred)
    return score

# 社保为x[18] 负相关
def forest(X_train, X_test, y_train, y_test,estimatorsnum):


    clf = RandomForestClassifier(n_estimators=estimatorsnum)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print('forest:')
    print(classification_report(y_test, pred))
    score = metrics.f1_score(y_test, pred, average='macro')

    #def getpr(y_true, y_pred):
    return score
def adaboost(X_train, X_test, y_train, y_test):


    clf = AdaBoostClassifier(n_estimators=100)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print('adaboost:')
    print(classification_report(y_test, pred))

def gdc(X_train, X_test, y_train, y_test):

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth = 1, random_state = 0).fit(X_train, y_train)
    pred = clf.predict(X_test)
    print('gdc:')
    print(classification_report(y_test, pred))
def regressn(X_train, X_test, y_train, y_test):
    #reg = sm.OLS.LinearRegression()
    reg=sm.OLS(y_train,X_train).fit()
    #print(reg.coef_)
    print(reg.summary())

def mlp(X_train, X_test, y_train, y_test):

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (5, 2), random_state = 1)
    clf.fit(X_train, y_train)
    MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                  beta_1=0.9, beta_2=0.999, early_stopping=False,
                  epsilon=1e-08, hidden_layer_sizes=(5, 2),
                  learning_rate='constant', learning_rate_init=0.001,
                  max_iter=200, momentum=0.9, n_iter_no_change=10,
                  nesterovs_momentum=True, power_t=0.5, random_state=1,
                  shuffle=True, solver='lbfgs', tol=0.0001,
                  validation_fraction=0.1, verbose=False, warm_start=False)
    pred = clf.predict(X_test)
    print('mlp:')
    print(classification_report(y_test, pred))

def tuningtree():
    para=1
    scmax=0
    bestpara=0
    scs=[]
    for para in range(1,100):

        sc=mytree(X_train2, X_test2, y_train2, y_test2,para)
        scs.append(sc)
        if sc>scmax:
            scmax=sc
            bestpara=para
    plt.plot(scs)
    plt.show()
    print('bestpara:')
    print(bestpara)
    print(scmax)
    return bestpara
def tuningforest():
    para=1
    scmax=0
    bestpara=0
    scs=[]
    for para in range(2,50):

        sc=forest(X_train2, X_test2, y_train2, y_test2,para)
        scs.append(sc)
        if sc>scmax:
            scmax=sc
            bestpara=para
    plt.plot(scs)
    plt.show()
    print('bestpara:')
    print(bestpara)
    print(scmax)
    return bestpara

def simplesvm(X_train, X_test, y_train, y_test):
    clf = svm.SVC(gamma='scale')
    clf.fit(X_train, y_train)
    pred=clf.predict(X_test)


    print('svm:')
    print(classification_report(y_test, pred))
#tuning()
#mytree(X_train1, X_test1, y_train1, y_test1,6)
def getpr(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

def getprsummary():
    pass
def getroc(y,preds):
    fpr, tpr, thresholds = metrics.roc_curve(y, preds)
    plt.plot(fpr, tpr, marker='o')
    plt.show()
def xgboost(X_train, X_test, y_train, y_test):


    clf = XGBClassifier(learning_rate=0.1,  # 默认0.3
    n_estimators=1100,  # 树的个数
    max_depth=3,
    min_child_weight=5,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',  # 逻辑回归损失函数
    nthread=5,  # cpu线程数
    scale_pos_weight=1,
    reg_alpha=1e-05,
    reg_lambda=2,
    seed=27)  # 随机种子

    clf.fit(X_train, y_train)
    data_predict = clf.predict(X_test)

    print(metrics.classification_report(y_test, data_predict))


X_train2, X_test2, y_train2, y_test2=pre_advanced()

# mytree(X_train1, X_test1, y_train1, y_test1,25)
# mytree(X_train2, X_test2, y_train2, y_test2,25)
# forest(X_train1, X_test1, y_train1, y_test1,126)
# forest(X_train2, X_test2, y_train2, y_test2,126)
# adaboost(X_train1, X_test1, y_train1, y_test1)
# adaboost(X_train2, X_test2, y_train2, y_test2)
# gdc(X_train1, X_test1, y_train1, y_test1)
# gdc(X_train2, X_test2, y_train2, y_test2)
# mlp(X_train1, X_test1, y_train1, y_test1)
# mlp(X_train2, X_test2, y_train2, y_test2)
#regressn(X_train1, X_test1, y_train1, y_test1)


#simplesvm(X_train1, X_test1, y_train1, y_test1)
#simplesvm(X_train2, X_test2, y_train2, y_test2)
#mytree(X_train2, X_test2, y_train2, y_test2,tuningtree())
#forest(X_train2, X_test2, y_train2, y_test2,tuningforest())
#forest(X_train2, X_test2, y_train2, y_test2,126)
xgboost(X_train2, X_test2, y_train2, y_test2)
#dataset_x=dataset['角色','子女','年龄','性别','年收入','家庭年收入','工作','主要出行工具是否为私家车(1：是 2：否)','是否赡养父母(1是 2否)','家庭成员是否有社保(1有 2无)','疾病']
##gittest