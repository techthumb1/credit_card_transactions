import os
import pickle
import numpy as np
import pandas as pd
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
import sys
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

# Save the model
pickle.dump(model, open('model.pkl', 'wb'))

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Predict
print(model.predict([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165,
166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332
]]))




#print(model.get_accuracy())
#print(model.get_confusion_matrix())
#print(model.get_classification_report())
#print(model.get_roc_auc_score())
#print(model.get_precision_recall_curve())
#model.save_model()
## Load the model
#model.load_model()
#print(model.predict(X_test))
#print(model.predict_proba(X_test))

# Path: WebApp/app.py
