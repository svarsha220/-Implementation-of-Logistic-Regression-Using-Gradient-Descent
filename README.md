## EX6:Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python packages
2. Read the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value

## Program:

```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: VARSHA S
RegisterNumber:  212222220055
*/

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv('Placement_Data_Full_Class.csv')
dataset.head()

dataset.info()

dataset = dataset.drop('sl_no', axis=1);
dataset.info()

dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

x = dataset.iloc[:,:-1]
x

y=dataset.iloc[:,-1]
y

import numpy as np
theta = np.random.rand(x.shape[1])
y=y

def sigmoid(z):
  return 1/(1+np.exp(-z))

def loss(theta, X, y):
  h = sigmoid(X.dot(theta))
  return npm.sum(y*np.log(h) + (1-y)*np.log(1-h))

def gradient_descent(X, y, theta, alpha, iterations):
  m = len(y)
  for i in range(iterations):
    h = sigmoid(X.dot(theta))
    gradient = X.T.dot(h-y) / m
    theta -= alpha * gradient
  return theta

theta = gradient_descent(x, y, theta, 0.01, 1000)

def predict(theta, X):
  h = sigmoid(X.dot(theta))
  y_pred = np.where(h >= 0.5, 1, 0)
  return y_pred 

y_pred = predict(theta, x)

accuracy = np.mean(y_pred == y)
print("Accuracy:", accuracy)
print("Predicted:\n",y_pred)
print("Actual:\n",y.values)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print("Predicted Result:",y_prednew)
```

### Head:
![Screenshot 2024-10-09 103951](https://github.com/user-attachments/assets/e3aad698-af06-43a4-bdc5-8a34bb88ca6a)


### Info:
![Screenshot 2024-10-09 104012](https://github.com/user-attachments/assets/8820330f-7e1e-48da-86d6-d317a106fbbb)


### Info:
![Screenshot 2024-10-09 104018](https://github.com/user-attachments/assets/2ce27a91-ad3d-4ab7-9fee-317e15627700)


### Changing into category:
![Screenshot 2024-10-09 104023](https://github.com/user-attachments/assets/25000505-014a-4957-a3f0-9dcdf86b0cfa)


### Changing into category codes:
![Screenshot 2024-10-09 104032](https://github.com/user-attachments/assets/5ec2f02d-e760-40c4-9f1c-f2eaf3de176e)


### Value of X:
![Screenshot 2024-10-09 104041](https://github.com/user-attachments/assets/a002ddd8-8164-4253-a6f5-00c8adc5b0f7)


### Value of Y:
![Screenshot 2024-10-09 104049](https://github.com/user-attachments/assets/055eb2cb-d880-4e0d-85f9-ee501596ab6c)


### Predicted Value:
![Screenshot 2024-10-09 113538](https://github.com/user-attachments/assets/46ce7d7d-1642-4587-a54f-e769c4b96c51)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

