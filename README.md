# Pseudo-Shapley-Value
A simple python program that calculates impacts of features (pseudo-Shapley values) for machine learning algorithms build in python.
This is simply a function written in python that takes in a dataframe, a machine learning model, and a specefied number of iterations, 
as arguments. The code returns a bar plot with the column names on the x-axis and corresponding variable impacts on the y-axis. This
can be used to determine which independent variables are important in a machine learning algorthm. This is process is model agnostic
thus can be applied to almost any machine learning algorith (ie neural networks, linear regression, random forest, XG boosted trees 
etc.) 
