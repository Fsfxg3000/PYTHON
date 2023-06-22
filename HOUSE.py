import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sb
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
import numpy
df = pd.read_csv("C:/Users/admin\Downloads/archive (2).zip")
print(df)

df = pd.read_csv("C:/Users/admin\Downloads/archive (2).zip")
print(df)
print(df.head(10))
print(df.tail())
print(df.columns.values)
print(df.describe())


###############################################################
#preparing X & Y
X= df.drop('RM',axis=1)
X= X.drop('LSTAT',axis=1)
Y=df['LSTAT']
print(X)
print(Y)

###############################################################
'''
# Feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures=SelectKBest(score_func=chi2,k='all')
fit = bestfeatures.fit(X,Y)                                      #training the model
dfscores = pd.DataFrame(fit.scores_)                             #storing the scores
dfcolumns = pd.DataFrame(X.columns)                              #storing coloumns
featuresScores = pd.concat([dfcolumns,dfscores],axis=1)          #concat scores and coloumns
featuresScores.columns = ['Survived','Scores']                   #giving lables to scores and columns

print(featuresScores)
'''
##############################################################################33
#Numerical to Categorical
df['PTRATIO']=pd.cut(df['PTRATIO'],2,labels=['0','1'])
df['LSTAT']=pd.cut(df['LSTAT'],2,labels=['0','1'])
df['MEDV']=pd.cut(df['MEDV'],2,labels=['0','1'])
df['RM']=pd.cut(df['RM'],2,labels=['0','1'])


print(df)
################################################################################

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()

le.fit(df['PTRATIO'])
df['PTRATIO']=le.transform(df['Age'])

le.fit(df['MEDV'])
df['MEDV']=le.transform(df['MEDV'])

le.fit(df['RM'])
df['RM']=le.transform(df['RM'])

le.fit(df['LSTAT'])
df['LSTAT']=le.transform(df['LSTAT'])


######################################################

#Dealing with null values
print(df)
print("Number of null values")
print(df.isnull().sum)
print(df.isnull().sum())

print("Number of not null")
print(df.notnull().sum)
print(df.notnull().sum())

m = df
X= m.drop("MEDV",axis = 1)
X = X.drop("LSTAT",axis=1)
X = X.drop("RM",axis=1)
X = X.drop("PTRATIO",axis=1)

Y =m["RM"]