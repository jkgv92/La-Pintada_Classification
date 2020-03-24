#!/home/programacion4/.linuxbrew/bin/python3
#PBS -N Pintada
#PBS -o Pintadaout.txt
#PBS -e Pintadaerr.txt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv('Datos originales_conectividad.csv')

# Select columns with characteristics
df = df.iloc[:, 2:]
x = df.values[:, 5:21]  # returns a numpy array

# Normalize dataset columns
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

# Restore dataset with normalized columns
df.iloc[:, 5:21] = x_scaled
df.head()

# Select desired characteristics
x1 = df.iloc[:, :].values
y1 = df.iloc[:, -1].values
# x2 = df.iloc[500:, :-1].values
# y2 = df.iloc[500:, -1].values

# Split train - test datasets
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.30)

# Show best characteristics
X_clf_new = SelectKBest(score_func=chi2, k=8).fit_transform(X_train[:, [8, 9, 11, 12, 13, 14, 17, 18]], y_train)
# # print(X_clf_new[0])
# print(X_train[0])

X_train, y_train = SMOTE().fit_resample(X_train, y_train)

# Predict with Random Forest
print('Prediction RF')
RF = RandomForestClassifier(bootstrap=False)
RF.fit(X_train[:, [8, 9, 11, 12, 13, 14, 17, 18]], y_train)
pred_RF = RF.predict(X_test[:, [8, 9, 11, 12, 13, 14, 17, 18]])
print(RF.feature_importances_)
print(classification_report(y_test, pred_RF))

# Select best characteristics


# Show best k value for KNN classifier
error = []
for i in range(1, 150):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train[:, [8, 9, 11, 12, 13, 14, 17, 18]], y_train)
    pred_i = knn.predict(X_test[:, [8, 9, 11, 12, 13, 14, 17, 18]])
    confusion = confusion_matrix(y_test, pred_i)
    error.append(np.mean(pred_i != y_test))

# plot k values against error
plt.figure(figsize=(12, 6))
plt.plot(range(1, 150), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
# plt.show()
k = (error.index(min(error))) + 1


# Predict with KNN
print('Prediction KNN')
KNN = KNeighborsClassifier(n_neighbors=k)
KNN.fit(X_train[:, [8, 9, 11, 12, 13, 14, 17, 18]], y_train)
y_pred = KNN.predict(X_test[:, [8, 9, 11, 12, 13, 14, 17, 18]])
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Predict with SVM
print('Prediction SVM')
svm = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001,
          cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
svm.fit(X_train[:, [8, 9, 11, 12, 13, 14, 17, 18]], y_train)
pred_svm = svm.predict(X_test[:, [8, 9, 11, 12, 13, 14, 17, 18]])
confusion_svm = confusion_matrix(y_test, pred_svm)
print(classification_report(y_test, pred_svm))



# Create dataframes with Xtrain and Xtest
labels_df = df.columns.values
X_train_pd = pd.DataFrame(data=X_train[:, :],  # values
                          columns=labels_df)  # 1st row as the column names
X_test_pd = pd.DataFrame(data=X_test[:, :],  # values
                         columns=labels_df)  # 1st row as the column names

# Export dataframes as excel sheets
X_test_pd.to_excel("X_test.xlsx")
X_train_pd.to_excel("X_train.xlsx")

# Export prediction vector
y_pred.tofile('y_pred.csv', sep=',', format='%10.5f')
