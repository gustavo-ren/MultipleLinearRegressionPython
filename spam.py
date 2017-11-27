import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("50_Startups.csv")

#Independent variables matrix creation
X = dataset.iloc[:, :-1].values
#Dependent varibles vector creation
y = dataset.iloc[:, 4].values

#Encoding the categorial State values
labelEncoderX = LabelEncoder()
X[:, 3] = labelEncoderX.fit_transform(X[:, 3])
hotEncoder = OneHotEncoder(categorical_features=[3])
X = hotEncoder.fit_transform(X).toarray()

X = X[:, 1:]

#Spliting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

regression = LinearRegression()
regression.fit(X_train, y_train)

#Predicting values
y_pred = regression.predict(X_test)

#Build of the Backward Elimination
#Adding the bias values on the independent variables matrix
#needs to be manually added for OLS does not add it by default
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

#Matrix containing  all the independent variables to be optimized
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
#Ordinary Least Squares model object (OLS)
regressionOLS = sm.OLS(endog=y, exog=X_opt).fit()
print (regressionOLS.summary())

#Eliminating p-values higher than 0.05
X_opt = X[:, [0, 1, 3, 4, 5]]
#Ordinary Least Squares model object (OLS)
regressionOLS = sm.OLS(endog=y, exog=X_opt).fit()
print (regressionOLS.summary())

X_opt = X[:, [0, 3, 4, 5]]
#Ordinary Least Squares model object (OLS)
regressionOLS = sm.OLS(endog=y, exog=X_opt).fit()
print (regressionOLS.summary())

X_opt = X[:, [0, 3, 5]]
#Ordinary Least Squares model object (OLS)
regressionOLS = sm.OLS(endog=y, exog=X_opt).fit()
print (regressionOLS.summary())

X_opt = X[:, [0, 3]]
#Ordinary Least Squares model object (OLS)
regressionOLS = sm.OLS(endog=y, exog=X_opt).fit()
print (regressionOLS.summary())