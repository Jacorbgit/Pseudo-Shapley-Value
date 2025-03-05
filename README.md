# Pseudo-Shapley-Value
A python program that calculates impacts and effects of features (pseudo-Shapley values) for machine learning algorithms built in python.
This is simply a function written in python that takes in a dataframe, a machine learning model, and a specefied number of iterations, 
as arguments. The code returns a bar plot with the column names on the x-axis and corresponding variable impacts on the y-axis. This
can be used to determine which independent variables are important in a machine learning algorthm. This is process is model agnostic
thus can be applied to almost any machine learning algorithm (ie neural networks, linear regression, random forest, XG boosted trees 
etc.) 
