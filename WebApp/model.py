import os
import sys
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import lightgbm as lgb


# Get the data
cct1 = pd.read_csv('/Users/jasonrobinson/Documents/Data-Engineering-Credit-Card-Transactions/transactions.csv')

#Preprocess the data
cct1['Zip'] = cct1['Zip'].fillna(0)
cct1['Amount'] = cct1['Amount'].apply(lambda value: float(value.split("$")[1]))
cct1['Hour'] = cct1['Time'].apply(lambda value: int(value.split(":")[0]))
cct1['Minutes'] = cct1['Time'].apply(lambda value: int(value.split(":")[1]))
cct1.drop(['Time'], axis=1, inplace=True)
cct1['Merchant Name'] = cct1['Merchant Name'].astype("object")
cct1['Card'] = cct1['Card'].astype("object")
cct1['Use Chip'] = cct1['Use Chip'].astype("object")
cct1['MCC'] = cct1['MCC'].astype("object")
cct1['Zip'] = cct1['Zip'].astype("object")

for col in cct1.columns:
    col_type = cct1[col].dtype
    if col_type == 'object' or col_type.name == 'category':
        cct1[col] = cct1[col].astype('category')

# Split our dataset in X and Y
y = cct1['Is Fraud?']
X = cct1.drop(['Is Fraud?'],axis=1)

categorical_column_names = []
categorical_cols = []
for idx,col in enumerate(X.columns):
    col_type = X[col].dtype
    if col_type == 'object' or col_type.name == 'category':
        categorical_column_names.append(col)
        categorical_cols.append(idx)
    
categorical_column_names.append("Zip")
categorical_column_names.append("MCC")
categorical_column_names.append("Card")
categorical_column_names.append("Merchant Name")

categorical_names = {}
for feature in categorical_column_names:
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(X.loc[:, feature])
    X.loc[:, feature] = le.transform(X.loc[:, feature])
    categorical_names[feature] = le.classes_

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = lgb.LGBMClassifier()
model.fit(X_train, y_train,
          feature_name='auto',
          categorical_feature=categorical_column_names,
          eval_set=[(X_test, y_test)],
          eval_metric='auc',
          early_stopping_rounds=50)


y_pred=model.predict(X_test)
y_pred_prob=model.predict_proba(X_test)[:,1]

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
print("Classification Report: ", classification_report(y_test, y_pred))
print("ROC AUC Score: ", roc_auc_score(y_test, y_pred_prob))
print("Precision Recall Curve: ", precision_recall_curve(y_test, y_pred_prob))

# Save the model to disk
import pickle
clf_model = 'classification_model.sav'
pickle.dump(model, open(clf_model, 'wb'))


# Load the model
# Load the model from disk
loaded_model = pickle.load(open(clf_model, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)

# Return is fraud or not fraud
y_pred = loaded_model.predict(X_test)
print(y_pred)

# Return probability of fraud
y_pred_prob = loaded_model.predict_proba(X_test)[:,1]   
print(y_pred_prob)
