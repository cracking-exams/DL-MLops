#  3. Saving and Reusing a Machine Learning Model: In this assignment, you will train a
# machine learning model using a simple dataset and learn how to save and reuse the
# model without retraining.
# a. Tasks to Perform:
# i. Train a Model:
# ii. Save the Model:
# iii. Reuse the Model: 

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
model_v1 = KNeighborsClassifier(n_neighbors=6)
model_v1.fit(x_train,y_train)

y_pred1 = model_v1.predict(x_test)
acc1 = metrics.accuracy_score(y_test,y_pred1)


joblib.dump(model_v1,'model_v1.pkl')



# Loading and reusing the model 
loaded_model = joblib.load('model_v1.pkl')

y_pred = loaded_model.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy of the reused model:", acc)