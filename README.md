# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1. Start

Step 2. Import numpy as np

Step 3. Plot the points

Step 4. IntiLiaze thhe program

Step 5.End

## Program:
```
 /*
Program to implement the linear regression using gradient descent.
Developed by:PRAVEEN K
RegisterNumber:  212223230153
*/


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        
        #calculate predictions
        predictions = (X).dot(theta).reshape(-1,1)
        
        #calculate errors
        errors=(predictions - y ).reshape(-1,1)
        
        #update theta using gradiant descent
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
                                        
data=pd.read_csv("C:/classes/ML/50_Startups.csv")
data.head()

#assuming the lost column is your target variable 'y' 

X = (data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler = StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X)
print(X1_Scaled)

#learn modwl paramerers

theta=linear_regression(X1_Scaled,Y1_Scaled)

#predict target value for a new data
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## Output:
## DATA.HEAD()
![image](https://github.com/user-attachments/assets/65f05d72-cc8e-4476-88d5-f28b3fdad852)

## X VALUE:
![image](https://github.com/user-attachments/assets/e20b74f7-b4d3-414f-8d46-e3633a04a128)
## X1_SCALED VALUE:
![image](https://github.com/user-attachments/assets/aa245c28-6535-4538-b0c7-100c8710d091)
## PREDICTED VALUES:
![image](https://github.com/user-attachments/assets/51d9426b-f391-4ccd-b445-bf525f71e772)





## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
