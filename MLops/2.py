# 2. Model Training and Versioning using a Simple Dataset: In this assignment, you will
# work with a small dataset (e.g., the Iris dataset) to build and manage versions of a basic
# machine learning model.
# a. Tasks to Perform:
# i. Data Preparation:
# ii. Model Training:
# iii. Hyperparameter Tuning:
# iv. Record the results for comparison.
# v. Model Versioning: Save each trained model as a separate version using
# meaningful filenames (e.g., model_v1.pkl, model_v2.pkl).

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import joblib

iris = load_iris()

x=iris.data
y=iris.target #setosa, versicolor , virginica   
# 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
model_v1 = KNeighborsClassifier(n_neighbors=7)
model_v1.fit(x_train,y_train)

y_pred1 = model_v1.predict(x_test)
acc1 = metrics.accuracy_score(y_test,y_pred1)


joblib.dump(model_v1,'model_v1.pkl')

# Hyperparameter tunning
model_v2 = KNeighborsClassifier(n_neighbors=8)
model_v2.fit(x_train,y_train)

y_pred2 = model_v1.predict(x_test)
acc2 = metrics.accuracy_score(y_test,y_pred2)

print(f'Accuracy of Model 1:{acc1}')
print(f'Accuracy of Model 2:{acc1}')

joblib.dump(model_v1,'model_v2.pkl')
