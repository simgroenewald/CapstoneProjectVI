import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_boston

# Loads the dataset needed
boston_dataset = load_boston()
col_list = boston_dataset['feature_names']

#  Selects the LSTAT column as x values
x = pd.DataFrame(data=boston_dataset['data'], columns=col_list)['LSTAT'].to_frame()
x.columns = ['x']

# Selects the target list as y values
y = pd.DataFrame(data=boston_dataset['target'])
y.columns = ['y']

# Selects the x and y training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.10,
                                                    random_state=42,
                                                    shuffle=True)

# Train the Linear Regression model and plot a prediction
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
xx = np.linspace(0, 26, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy, c='g')

# Set the degree of the Polynomial Regression model
quadratic_featurizer = PolynomialFeatures(degree=5)

X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.transform(X_test)

# Trains and tests the regressor_quadratic model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, Y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

# Plot the graph
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')
plt.title('% Status vs House Price')
plt.xlabel('% Status')
plt.ylabel('House Price')
plt.grid(True)
plt.scatter(X_train, Y_train, s=10)
plt.show()
print(X_train)
print(X_train_quadratic)
print(X_test)
print(X_test_quadratic)
