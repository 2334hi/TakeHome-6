# David Huang | Data Science Bootcamp Take Home #6
#Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.linear_model import Lasso
from sklearn.linear_model import ridge_regression

import warnings 
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 101)
data = pd.read_csv("employee.csv")


#1 Preprocessing and Getting Predictions 

# Making predictions of an employee's salary based on their job years, hours per week, 
# and telecommute days per week, using a Linear Regression, which models the 
# dependent variable of salary to the independent variables selected. 

y = data['salary']
X = data.drop(columns=['salary'])

num_cols = ['job_years','hours_per_week','telecommute_days_per_week']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train[num_cols])
X_train[num_cols] = scaler.transform(X_train[num_cols])
reg = LinearRegression()

# reg.fit(X_train, y_train) // Appears to give an error, cannot convert a date string to float...?
reg.fit(X_train, y_train)


# 2 Computing Mean Absolute Error, and Mean Squared Error
MSE = mean_squared_error(y_train,reg.predict(X_train))/np.mean(y_train) # Cannot compute unless reg.fit is done

# Mean Abs. Error - Avg diff between calculated values and actual values
MAB = mean_absolute_error(y_train,reg.predict(X_train))

# 3 Ridge and Lasso Regression // TBD
reg2 = ridge_regression()

