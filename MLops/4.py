# 4. Creating a Reproducible ML Pipeline using Jupyter and Virtual Environment:
# a. Tasks to Perform:
# i. Set up a virtual environment using venv or conda.
# ii. Install necessary libraries like scikit-learn, pandas, matplotlib.
# iii. Create a Jupyter notebook that:
# iv. Loads a dataset (e.g., Titanic, Wine).
# v. Performs data preprocessing.
# vi. Trains a simple model.

# vii. Save the notebook and environment dependencies
# (requirements.txt).
# viii. Share the notebook and environment setup on GitHub for others
# to reproduce.



# steps
# 1. create a folder on desktop
# 2. open cmd in that folder & run the following commands in cmd
#     ->python -m venv mlops
#     ->mlops\Scripts\activate   
#     ->pip install jupyter numpy scikit-learn pandas matplotlib seaborn
#     ->pip freeze > requirements.txt
#     ->ipython kernel --install --user --name=mlops
# 3. Open jupyter notebook and select the kernel with name mlops
# 4. copy paste the below code 
# 5. Put all the files to github


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


iris = load_iris()

x=iris.data
y=iris.target #setosa, versicolor , virginica   
# 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
model_v1 = KNeighborsClassifier(n_neighbors=6)
model_v1.fit(x_train,y_train)

y_pred1 = model_v1.predict(x_test)
acc1 = metrics.accuracy_score(y_test,y_pred1)
print(acc1)