import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing



df = pd.read_csv('Datos originales_conectividad.csv')


df=df.iloc[:,2:]
x = df.values[:,5:21] #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df.iloc[:,5:21]=x_scaled
df.head()

x1 = df.iloc[:,[5,7,11,12,13]].values
y1 = df.iloc[:, -1].values
#x2 = df.iloc[500:, :-1].values
#y2 = df.iloc[500:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.30)

###Train Set
X_train=x1[:,[5,7,11,12,13]]
y_train=y1
###Test Set
X_test=x2[:,[5,7,11,12,13]]
y_test=y2


from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


X_clf_new = SelectKBest(score_func=chi2,k=5).fit_transform(X_train,y_train)
print(X_clf_new[:5])

print(X_train[:5])


RF = RandomForestClassifier(bootstrap=True)
RF.fit(X_train, y_train)
pred_RF = RF.predict(X_test)
print(RF.feature_importances_)
print(classification_report(y_test, pred_RF))

error = []
X_train, y_train = SMOTE().fit_resample(X_train, y_train)
for i in range(1, 150):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    confusion=confusion_matrix(y_test, pred_i)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 150), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

k=(error.index(min(error)))+1
print(k)

classifier = KNeighborsClassifier(n_neighbors=k)

X_train, y_train = SMOTE().fit_resample(X_train, y_train)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
