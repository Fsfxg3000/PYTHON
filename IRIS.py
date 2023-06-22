import pandas as pd
from sklearn.datasets import load_iris
irs =load_iris()

print(irs)
print(irs.key)
print(irs.data)
print(irs.traget)
print(irs.feature_name)
print(irs.traget_name)
df=pd.read_csv("D:/CDSA/iris.csv")
print(df)
print(df.head(10))
print(df.tail(1))
print(df.columns.values)
print(df.describe())


import pandas as pd

df = pd.read_csv("C:/Hello/Iris.csv")

X = df.drop('Id',axis=1)
X = X.drop('Species',axis=1)
Y = df['Species']
print(X)
print(Y)

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeature = SelectKBest (score_func=chi2,k='all')
fit =bestfeature.fit(X,Y)
dfscores =pd.DataFrame(fit.scores_)
dfcolumns =pd.DataFrame(X.columns)
featureScores = pd.concat ([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']

print(featureScores)

import pandas as pd

df = pd.read_csv("C:/Hello/Iris.csv")

X = df.drop('Id',axis=1)
X = X.drop('Species',axis=1)
Y = df['Species']
print(X)
print(Y)

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeature = SelectKBest (score_func=chi2,k='all')
fit =bestfeature.fit(X,Y)
dfscores =pd.DataFrame(fit.scores_)
dfcolumns =pd.DataFrame(X.columns)
featureScores = pd.concat ([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']

print(featureScores)

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)

feat_importance = pd.Series(model.feature_importances_,index=X.columns)
feat_importance.nlargest(4).plot(kind='barh')
plt.show()

df['SepalLengthCm']=pd.cut(df['SepalLengthCm'],3,labels=['0','1','2'])
df['SepalWidthCm']=pd.cut(df['SepalWidthCm'],3,labels=['0','1','2'])
df['PetalLengthCm']=pd.cut(df['PetalLengthCm'],3,labels=['0','1','2'])
df['PetalWidthCm']=pd.cut(df['PetalWidthCm'],3,labels=['0','1','2'])

print(df)

import pandas as pd

df = pd.read_csv("C:/Hello/Iris.csv")
print(df)
print("number of null value: ")
print(df.isnull().sum)
print(df.notnull().sum())

print("number of the null")
print(df.isnull().sum)
print(df.notnull().sum())

df['SepalLengthCm'].fillna((df['SepalLengthCm'].mean()),inplace=True)
df['PetalLengthCm'].fillna((df['PetalLengthCm'].mean()),inplace=True)
print(df.isnull().sum())

print(df)

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X ,Y = ros.fit_resample(X,Y)

from imblearn.over_sampling import SMOTE
sms = SMOTE(random_state=0)
X ,Y = sms.fit_resample(X,Y)