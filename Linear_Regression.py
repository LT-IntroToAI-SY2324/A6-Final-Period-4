import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("part1-linear-regression/blood_pressure_data.csv")
print(data)
x = data["Year"].values
y = data["Stock Price"].values

x = x.reshape(-1,1)

model=LinearRegression().fit(x,y)
# Find the coefficient, bias, and r squared values. 
# Each should be a float and rounded to two decimal places. 
coef = round(float(model.coef_), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(x, y)
print(coef, intercept, r_squared)

# Print out the linear equation and r squared value
print(f"Model's Linear Equation: y = {coef}x + {intercept}")
print(f"R Squared value: {r_squared}")

x_predict = 43
# plug that value into your model
prediction = model.predict([[x_predict]])
print(x_predict)
print(f"Prediction when x is {x_predict}: {prediction}")


# Create the model in matplotlib and include the line of best fit


# sets the size of the graph
plt.figure(figsize=(6,4))

# creates a scatter plot of originial data in purple
# and the predicted data in blue
plt.scatter(x,y, c="purple")
plt.scatter(x_predict, prediction, c="blue")



# plot the line of best fit in red and label the line
plt.plot(x, coef*x + intercept, c="r", label="Line of Best Fit")