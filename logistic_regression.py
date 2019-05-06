import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


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


best_score = 0
for penalty in ['l1', 'l2']:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]: # inverse of regularization
        logreg = LogisticRegression(class_weight='balanced',  penalty=penalty, C=C, solver='liblinear')
        logreg.fit(train_, y_train)
        score = logreg.score(val_, y_val)
        if score > best_score:
            best_score = score
            # print(best_score)
            best_parameters = {'C': C, 'penalty': penalty}

logreg = LogisticRegression(**best_parameters)
logreg.fit(train_, y_train)
accuracy = logreg.score(val_, y_val)
print("accuracy:", accuracy)

# Logistic Regression model
y_pred_final=logreg.predict_proba(test_)[:,1]

submission = pd.DataFrame({"id": test["id"],"target": y_pred_final})
submission.to_csv('submission_log.csv', index=False)