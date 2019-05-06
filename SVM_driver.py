import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Preparing the data
X = train.drop(['id','target'],axis=1)
y = train['target']
X_test = test.drop(['id'],axis=1)


# Splitting the dataset in training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, shuffle = None, random_state = 0)
scaler = StandardScaler()
train_ = scaler.fit_transform(X_train)
val_ = scaler.fit_transform(X_val)
test_ = scaler.transform(X_test)

svc = SVC()
param_grid = [{'kernel':['linear'], 'C': [0.01,100], 'probability':[True]}]

gs = GridSearchCV(svc, param_grid, cv = 5)
gs.fit(X_val, y_val)

print(gs.best_params_)
svc = SVC(**gs.best_params_)
svc.fit(train_, y_train)
yP = svc.predict(X_val)

print(accuracy_score(y_val, yP))
