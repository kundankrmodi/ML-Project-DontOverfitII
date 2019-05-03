import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Preparing the data
X = train.drop(['id','target'],axis=1)
y = train['target']
X_test = test.drop(['id'],axis=1)


# Splitting the dataset in training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.6, shuffle = None, random_state = 0)
scaler = StandardScaler()
train_ = scaler.fit_transform(X_train)
val_ = scaler.fit_transform(X_val)
test_ = scaler.transform(X_test)



svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_val)

# get the accuracy
print (accuracy_score(y_val, y_pred))

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_val,y_pred))
print(classification_report(y_val,y_pred))