# Pseudo-Shapley-Value
This is a python program I wrote that calculates the strength of influence of features (pseudo-Shapley values) for machine learning algorithms built in python.
This is a program that takes in a dataframe, a machine learning model, and a specified number of iterations, as arguments. The code returns a bar plot with the column names on the x-axis and corresponding feature strength of effects on the y-axis. This can be used to determine which independent variables are important in a machine learning algorithm and which ones are less influential and may be irrelevant. This is process is model agnostic thus can be applied to almost any machine learning algorithm (ie neural networks, linear regression, random forest, XG boosted trees 
etc.) 
CONTACT: For any questons feel free to contact me directly at 702-773-4132
# Explanation of Files in this Repository:

## Making_Sample_Data.py

Making a sample dataset to test the program on (lines 11-43). I then make a model to predict the independent variable (Y) (lines 46). This program is model agnostic so this OLS model can be substituted with a neural network, random forest, XGboosted tree, or any other predictive algorithm. 

## sample_data_regression_summary.png

This is the summary of the model created in [Making_Sample_Data.py]. I made the 3 of the five independent variables (X1, X2, X3)  statistically significant (aka informative) hence their small p-values. If I wrote the program correctly these three variables should have large values since they are literally designed to have a large strength of influence on the model’s predictions. The other two variables (X4, X5) should have small values because the model does not take their influence into consideration when making predictions. 

## Shapley_Value_function.py

This is the program that takes in a model and produces a bar plot that conveys the strength of influence of each variable. Variables with higher strengths of influence are interpreted to be more informative, important, and overall more relevant to the model’s predictions while variables with lower strength of influence are interpreted to be less informative (and oftentimes can be removed from the model altogether). 

## Feature_Effects_on_output.png

This visual shows which variables the program found to be relevant. As we see X1, X2, and X3 have a very strong influence on the model’s output while X4 and X5 do not. This agrees with how the data was initially designed. 
