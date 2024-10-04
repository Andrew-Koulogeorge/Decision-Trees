from sklearn import datasets
from sklearn.model_selection import train_test_split
from DT import DecisionTree
import numpy as np

# load in new dataset
data = datasets.load_breast_cancer()
X,y = data.data, data.target
# split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15213)

# make instance of DT 
h = DecisionTree()
h.fit(X_train, y_train)
y_hat = h.predict(X_test)
print(np.sum(y_hat == y_test) / len(y_hat))

# compute performance