import pandas as pd
import numpy as np

#Function that uses gradient descent to update all theta values
def gradientDescent(theta_values, examples):
    #Calculate the dot product of the theta values and all of the examples
    #Which gives us a single value for each example (A 1598 x 1 matrix)
    sum = np.dot(examples, theta_values)
    #Take that dot product and subtract it by the class labels
    sum = sum - y
    #Take the dot product of the transpose of all the examples and the matrix of singular values for each example
    #This gives us a 12 x 1 matrix that holds the gradient for each feature
    gradient = np.dot(examples.transpose(), sum)
    #Calculate the average for each gradient
    average = gradient * (1/m)
    #Calculate the new theta values by taking the old theta value and subtracting it by the average gradient for 
    # that feature * the alpha value
    theta_values = theta_values - (alpha * average)
    return theta_values

#Function that calculates the mean squared error
def meanSquaredError(theta_values, examples):
    #Calculate the dot product of all the axamples and the theta values
    #This gives us a 1598 X 1 matrix (One value for each example)
    sum = np.dot(examples, theta_values)

    #Take each value from the new matrix and subtract it by the corresponding target value for that example
    #Then square that result
    sum = (sum - y) ** 2

    #Add all 1588 values together
    sum = sum.sum()

    #Calculate the average of those values and that will give us the MSE
    MSE = (1/m)*sum
    
    return MSE

#Function that performs linear regression on a dataset
def linearRegression(theta_values, examples):
    #Update the theta values 5000 times
    for i in range(5000):
        theta_values = gradientDescent(theta_values, examples)
    #Display the mean squared error once we have the final theta values
    print("Mean Squared Error: ", meanSquaredError(theta_values, examples))
    print("Weight Values: ", theta_values)


#Set alpha value
alpha = .001

#Read in the wine data
data = pd.read_csv('winequality-red.csv')

#Create dataframe for the examples and their features
x = data.iloc[0:,0:11]

#Create dataframe for the class label and then convert the dataframe into a numpy array
y = data.iloc[0:, 11]
y = y.to_numpy()


#Normalize all the values of the dataframe based on their feature
for column in x.iloc[:,:]:
    x[column] = (x[column] - x[column].min())/(x[column].max() - x[column].min())
#Add an initialization feature that is just equal to 1
x.insert(0, 'Feature0', 1)

#Turn the dataframe of examples into a numpy array
x_array = x.to_numpy()

#Initialize theta values (one for every feature)
theta_values = np.random.uniform(-1, 1, 12)

#Find the number of examples
m = len(x.index)

#Call the linear regression function the wine dataset
linearRegression(theta_values, x_array)


