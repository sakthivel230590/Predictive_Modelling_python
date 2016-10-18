import pandas as pd
import numpy as np
import seaborn as sns
bank = pd.read_csv("C:/Users/sakthivel/Desktop/Project/bank-full.csv",sep=";")
bank.head() # head of the bank
bank.tail()
print(bank.head())
bank.describe()
print(bank.describe())

bank['job'].describe()
bank.isnull().sum() # to check nas in any column
pd.value_counts(bank["marital"])# no of unique values with frequency
bank.dtypes # gives the column type of the dataset

bank.groupby("marital").mean()# group by marital the whole data set by mean
bank.groupby("marital").median()
bank.groupby("marital").sum()
bank.education.unique()
bank.education.unique()# returns no of unique values in the column
np.unique(bank.education)#returns no of unique values in the column
np.unique(bank.job)
pd.crosstab(bank.marital,bank.y)# group by marital in the row and bank.y in the column
bank.shape # returns no of rows and columns in the dataset
pd.DataFrame(bank.columns) # returns all the columns in the bank with their indexes
bank.rename()
y=bank.groupby("marital")# group by marital the whole data set
z=y.describe()# outcome as mean ,median,count for all numerical values
z
bank.dropna()# drop na
bank['age'].fillna(bank['age'].mean(), inplace=True)#fill na
bank["job"].value_counts()
bank.marital.unique()
bank["pdays"].value_counts() # give the count of all the unique values in that column
bank.marital[bank["marital"]=='single']=0 # recode
bank.marital[bank["marital"]=='divorced']=1# recode
bank.marital[bank["marital"]=='married']=2# recode
bank["contact"].value_counts()
bank.contact[bank["contact"]=='cellular']=0# recode
bank.contact[bank["contact"]=='unknown']=0# recode
bank.contact[bank["contact"]=='telephone']=1# recode
bank["default"].value_counts()
bank.default[bank["default"]=='no']=0
bank.default[bank["default"]=='yes']=1
bank["housing"].value_counts()
bank.housing[bank["housing"]=='no']=0
bank.housing[bank["housing"]=='yes']=1
bank["month"].unique()
bank.month[bank["month"]=='jan']=1
bank.month[bank["month"]=='feb']=2
bank.month[bank["month"]=='mar']=3
bank.month[bank["month"]=='apr']=4
bank.month[bank["month"]=='may']=5
bank.month[bank["month"]=='jun']=6
bank.month[bank["month"]=='jul']=7
bank.month[bank["month"]=='aug']=8
bank.month[bank["month"]=='sep']=9
bank.month[bank["month"]=='oct']=10
bank.month[bank["month"]=='nov']=11
bank.month[bank["month"]=='dec']=12
bank["poutcome"].unique()
bank.poutcome[bank["poutcome"]=='unknown']=0
bank.poutcome[bank["poutcome"]=='failure']=1
bank.poutcome[bank["poutcome"]=='other']=2
bank.poutcome[bank["poutcome"]=='success']=3
bank["job"].value_counts()
bank.job[bank["job"]=='blue-collar']=1
bank.job[bank["job"]=='management']=2
bank.job[bank["job"]=='technician']=3
bank.job[bank["job"]=='admin.']=4
bank.job[bank["job"]=='services']=5
bank.job[bank["job"]=='retired']=6
bank.job[bank["job"]=='self-employed']=7
bank.job[bank["job"]=='management']=8
bank.job[bank["job"]=='entrepreneur']=9
bank.job[bank["job"]=='unemployed']=10
bank.job[bank["job"]=='housemaid']=11
bank.job[bank["job"]=='student']=12
bank.job[bank["job"]=='unknown']=13
bank["education"].value_counts()
bank.education[bank["education"]=='secondary']=1
bank.education[bank["education"]=='tertiary']=2
bank.education[bank["education"]=='primary']=3
bank.education[bank["education"]=='unknown']=4

bank.loan[bank["loan"]=='no']=0
bank.loan[bank["loan"]=='yes']=1

bank['age'].hist() # histogram

bank.boxplot(column='age', by = 'marital') # boxplot
bank.apply(lambda x: sum(x.isnull()),axis=0) # sum of all the na in the datset in different columns where 0 means columns


bank['age']=np.log(bank.age)#apply log on a certain column
bank['age'].hist()
bank['balance'].hist()


from sklearn import preprocessing
bank['balance']=preprocessing.scale(bank.balance)# apply scale on certain column

bank['duration'].hist()
bank['duration']=preprocessing.scale(bank.duration)
bank.dtypes

# change datatype.....

bank[['job', 'marital','education','default','housing','loan','contact','month','poutcome','y']] = bank[['job', 'marital','education','default','housing','loan','contact','month','poutcome','y']].astype(int)
bank.dtypes
bank["y"].unique()
bank.y[bank["y"]=="yes"]=1
bank.y[bank["y"]=="no"]=0
corr = bank.corr() # find the correlation
cmap = sns.diverging_palette(h_neg=600, h_pos=600, s=90, l=90, as_cmap=True, center="light")
sns.clustermap(bank.corr(), figsize=(10, 10), cmap=cmap)
bank.head()

msk = np.random.rand(len(bank)) < 0.8
train = bank[msk]
test=bank[~msk]
train.shape
test.shape
pd.DataFrame(bank.columns)
x=train.ix[:,0:16]
y=train['y']

xtest=test.ix[:,0:16]
ytest=test["y"]
from sklearn import tree
model1 = tree.DecisionTreeRegressor()
model1.fit(x, y)
predicted= model1.predict(xtest)

df_confusion = pd.crosstab(ytest, predicted)

#col_0   0.0  1.0
#y               
#0      7485  561
#1       543  503

from sklearn.ensemble import RandomForestClassifier
model2= RandomForestClassifier(n_estimators=1000)
model2.fit(x, y)
predicted2= model2.predict(xtest)

df_confusion2 = pd.crosstab(ytest, predicted2)

#col_0     0    1
#y               
#0      7831  215
#1       616  430

from sklearn import svm
clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(x,y)
predicted3=clf.predict(xtest)
df_confusion3 = pd.crosstab(ytest, predicted3)


from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(x, y)
predicted4= model3.predict(xtest)
df_confusion4 = pd.crosstab(ytest, predicted4)

#col_0     0    1
#y               
#0      7268  796
#1       528  484
from sklearn.ensemble import GradientBoostingClassifier
clf2 = GradientBoostingClassifier(n_estimators=1000, learning_rate=.8, max_depth=1)
clf2.fit(x, y)
predicted5= clf2.predict(xtest)
df_confusion5 = pd.crosstab(ytest, predicted5)
#col_0     0    1
#y               
#0      7848  216
#1       658  354
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
logistic.fit(x,y)
predicted6=logistic.predict(xtest)
df_confusion6 = pd.crosstab(ytest, predicted6)
#col_0     0    1
#y               
#0      7835  214
#1       751  325