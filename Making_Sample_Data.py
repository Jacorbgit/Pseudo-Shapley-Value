import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Below we create a sample dataframe and perform a linear regression
# to demonstrate the algorithm. 

# Set random seed for reproducibility
np.random.seed(42)

# Generate 100 samples
n = 100

# Create 3 significant features (with noise)
X1 = np.random.normal(0, 1, n)  # Significant feature 1
X2 = np.random.normal(5, 2, n)  # Significant feature 2
X3 = np.random.normal(10, 3, n)  # Significant feature 3

# Create 2 non-significant features (purely random noise)
X4 = np.random.normal(20, 5, n)  # Non-significant feature 1
X5 = np.random.normal(50, 10, n)  # Non-significant feature 2

# Create the dependent variable (Y) as a linear combination of X1, X2, and X3 with some noise
Y = 3 * X1 + 2 * X2 - 1.5 * X3 + np.random.normal(0, 2, n)

# Create a DataFrame
data = pd.DataFrame({
    'X1': X1,
    'X2': X2,
    'X3': X3,
    'X4': X4,
    'X5': X5,
    'Y': Y
})

# Split the data into features (X) and target (Y)
X = data[['X1', 'X2', 'X3', 'X4', 'X5']]
Y = data['Y']

# Add a constant to the independent variables (for the intercept in the model)
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(Y, X).fit()

# Get the summary of the regression
summary = model.summary()

# Print the summary
print(summary)
