import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split

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
x1 = df.iloc[:, [5, 7, 11, 12, 13]].values
y1 = df.iloc[:, -1].values
# x2 = df.iloc[500:, :-1].values
# y2 = df.iloc[500:, -1].values

# Split train - test datasets
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.30)

# Show best characteristics
X_clf_new = SelectKBest(score_func=chi2, k=5).fit_transform(X_train, y_train)
print(X_clf_new[:5])
print(X_train[:5])


# Predict with Random Forest
RF = RandomForestClassifier(bootstrap=True)
RF.fit(X_train, y_train)
pred_RF = RF.predict(X_test)
print(RF.feature_importances_)
print(classification_report(y_test, pred_RF))

# Select best characteristics


# Show best k value for KNN classifier
error = []
X_train, y_train = SMOTE().fit_resample(X_train, y_train)
for i in range(1, 150):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    confusion = confusion_matrix(y_test, pred_i)
    error.append(np.mean(pred_i != y_test))

# plot k values against error
plt.figure(figsize=(12, 6))
plt.plot(range(1, 150), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
k = (error.index(min(error))) + 1
print(k)

# Predict with KNN
classifier = KNeighborsClassifier(n_neighbors=k)
X_train, y_train = SMOTE().fit_resample(X_train, y_train)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
