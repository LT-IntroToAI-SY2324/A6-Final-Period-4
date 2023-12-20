import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

'''
**********CREATE THE MODEL**********
'''

data = pd.read_csv("Coca_Cola_Data.csv")
x = data["Time (Year)"].values
y = data[" Annual Revenue (US dollars in Billions)"].values
print(x,y)
# Create your training and testing datasets:
xtrain, xtest, ytrain, ytest = train_test_split(x, y ,test_size = .2)
# Use reshape to turn the x values into 2D arrays:
xtrain = xtrain.reshape(-1,1)
print(xtrain, ytrain)
# Create the model
model = LinearRegression().fit(xtrain, ytrain)
# Find the coefficient, bias, and r squared values. 
# Each should be a float and rounded to two decimal places. 
coef = round(float(model.coef_), 2)
r_squared = model.score(xtrain, ytrain)
intercept = round(float(model.intercept_), 2)
# Print out the linear equation and r squared value:
print("Model's Linear Equation: y=",coef, "x=", intercept)
print("R Squared value:", r_squared)
'''
**********TEST THE MODEL**********
'''
# reshape the xtest data into a 2D array
print(xtest)
xtest = xtest.reshape(-1, 1)
print("this is the xtest", xtest)
# get the predicted y values for the xtest values - returns an array of the results
predict = model.predict(xtest)
# round the value in the np array to 2 decimal places
predict = np.around(predict, 2)

# Test the model by looping through all of the values in the xtest dataset
print("\nTesting Linear Model with Testing Data:")
for index in range(len(xtest)):
    actual = ytest[index]
    predicted_y = predict[index]
    x_coord = xtest[index]
print("x value:", float(x_coord), "Predicted Y Value: ", predicted_y, "Actual Y Value: ", actual)

'''
**********CREATE A VISUAL OF THE RESULTS**********
'''
plt.figure(figsize=(5,4))

plt.scatter(xtrain, ytrain, c="purple", label="Training Data")
plt.scatter(xtest, ytest, c="blue", label="Testing DAata")

plt.xlabel("Time (Year)")
plt.ylabel("Annual Revenue (US dollars in Billions)")
plt.title("Coca Cola Revenue by Year")
plt.plot(x, coef*x + intercept, c="r", label="Coca Cola Sales by Year")

plt.legend()
plt.show()