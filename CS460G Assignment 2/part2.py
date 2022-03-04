import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Function that adds more feature values to a dataframe that are higher order polynomials of the feature that is already there
def newOrder(order, dataframe):
    #Find the number of new features to add
    #This will always be the order polynomial that we are trying to achieve - one because we already have
    #a feature that is order one
    numNewFeatures = order - 1
    #We start adding features at an order of degree of 2 because we already have an order of one
    degree = 2
    #Add all the new higher order features
    for i in range(numNewFeatures):
        #Determine what the feature value will be by raising the order one feature to the degree that we are on
        dataframe[degree] = dataframe.iloc[:,1:2] ** degree
        degree += 1
    #Return the new dataframe
    return dataframe


#Function that uses gradient descent to update all theta values
def gradientDescent(theta_values, examples, y):
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
def meanSquaredError(theta_values, examples, y):
    #Calculate the dot product of all the axamples and the theta values
    #This gives us a 1598 X 1 matrix (One value for each example)
    sum = np.dot(examples, theta_values)

    #Take each value from the new matrix and subtract it by the corresponding target value for that example
    #Then square that result
    sum = (sum - y) ** 2

    #Add all 1598 values together
    sum = sum.sum()
    
    #Calculate the average of those values and that will give us the MSE
    MSE = (1/m)*sum
    
    return MSE

#Function that performs linear regression on a dataset
def regression(theta_values, examples, y):
    #Update the theta values 10000 times
    for i in range(10000):
        theta_values = gradientDescent(theta_values, examples, y)
    #Display the mean squared error once we have the final theta values
    print("Mean Squared Error: ", meanSquaredError(theta_values, examples, y))
    print("Weight Values: ", theta_values)
    return theta_values

#Function that takes in an array of theta values and an array of x values and creates function values
# based on polynomial regression
def createFunction(theta_values, x):
    ans = 0
    #Start the order of the polynomial at 0
    order = 0
    #Go through every theta value
    for value in theta_values:
        #Build the function values by multiplying the theta value by the x value raised to the order
        #of polynomial that we are on
        ans += (value * (x**order))
        #Raise the order of the polynomial for each theta value
        order += 1
    #Return the function values
    return ans

#Function that plot the data points and the polynomial regression function
def createPlot(dataframe, classLabel, theta_values, title, order):
    #Create initial scatterplot with the data points
    plt.scatter(dataframe.iloc[:, 1:2], classLabel.iloc[:, 1:2])
    #Create x values to use for polynomial regression function
    x = np.linspace(dataframe.iloc[:,1:2].min(), dataframe.iloc[:,1:2].max(), 100)
    #Create the polynomial regression function and get the corresponding values
    y = createFunction(theta_values, x)
    #Plot the function in the scope of the x values
    plt.plot(x,y,'g')
    #Make the title of the plot the dataset that we are using to make the polynomial regression
    order = str(order)
    plotTitle = title + " Order: " + order
    plt.title(plotTitle)
    #Add x and y labels
    plt.xlabel('Feature Value')
    plt.ylabel('Class Value')
    #Show the plot
    plt.show()

#BONUS - Create function that uses L2 Norm Regularizaton
def L2Norm(theta_values, examples, y):
    #Calculate the dot product of the theta values and all of the examples
    #Which gives us a single value for each example (A 1598 x 1 matrix)
    
    sum = np.dot(examples, theta_values)
    #Take that dot product and subtract it by the class labels
    sum = sum - y
    #Take the dot product of the transpose of all the examples and the matrix of singular values for each example
    #This gives us a 12 x 1 matrix that holds the gradient for each feature
    gradient = np.dot(examples.transpose(), sum)

    #Add the lambda value times the feature value
    gradient = gradient + (lambdaVal * theta_values)
    
    #Calculate the average for each gradient
    average = gradient * (1/m)
    #Calculate the new theta values by taking the old theta value and subtracting it by the average gradient for 
    # that feature * the alpha value
    theta_values = theta_values - (alpha * average)

    return theta_values


#Create a list of files to go through
files = ['synthetic-1.csv', 'synthetic-2.csv']
#Go through each file and calculate the MSE for each file with orders (2, 3, and 5)
for file in files:
    print("Filename: ", file)

    #Use an alpha value of .25 is the file is 'synthetic-1.csv'
    if file == 'synthetic-1.csv':
        alpha = .0025
    #Else use an alpha value of .001
    else:
        alpha = .025
    
    #Read in the file and create a dataframe
    data = pd.read_csv(file , header=None)

    #Create dataframe for the examples and their features
    x = data.iloc[0:, 0:1]

    #Add an initialization feature that is just equal to 1
    x.insert(0, 'Feature0', 1)

    #Create dataframe for the class label and then convert the dataframe into a numpy array
    y_data = data.iloc[:, 1]
    y = y_data.to_numpy()

    #Find the number of examples
    m = len(x.index)
    
    #List for the polynomial orders to use
    orders = [2, 3, 5]

    #Go through each polynomial order and calculate the MSE
    for order in orders:
        print("Order: ", order)

        #Initialize theta values (one for every feature)
        theta_values = np.random.uniform(-1, 1, order+1)
        
        #Add the new features based on the polynomial order
        x = newOrder(order, x)

        #Turn the dataframe of examples into a numpy array
        x_array = x.to_numpy()

        #Perform polynomial regression and gradient descent on the data to find the MSE
        theta_values = regression(theta_values, x_array, y)
        
        #Create Plot for each order polynomial
        #createPlot(x, data, theta_values, file, order)
    print("\n")


print("BONUS")
#BONUS - Use L2 Norm Regularization on both datasets with order 5 polynomial
for file in files:
    print("Filename: ", file)

    #Use an alpha value of .25 is the file is 'synthetic-1.csv'
    if file == 'synthetic-1.csv':
        alpha = .0025
    #Else use an alpha value of .001
    else:
        alpha = .025
    
    #Read in the file and create a dataframe
    data = pd.read_csv(file , header=None)

    #Create dataframe for the examples and their features
    x = data.iloc[0:, 0:1]

    #Add an initialization feature that is just equal to 1
    x.insert(0, 'Feature0', 1)

    #Create dataframe for the class label and then convert the dataframe into a numpy array
    y_data = data.iloc[:, 1]
    y = y_data.to_numpy()

    #Find the number of examples
    m = len(x.index)

    print("Order: 5")

    #Initialize theta values (one for every feature)
    theta_values = np.random.uniform(-1, 1, 6)
        
    #Add the new features based on the polynomial order
    x = newOrder(5, x)

    #Turn the dataframe of examples into a numpy array
    examples = x.to_numpy()

    #Initiliaze lambda value for L2 Norm Rgularization
    lambdaVal = .1

    #Update the theta values by using L2 Norm Regularization 10000 times
    for i in range(10000):
        theta_values = L2Norm(theta_values, examples, y)

    #Display the mean squared error and weight values once we have the final theta values
    print("Lambda Value: ", lambdaVal)
    print("Mean Squared Error: ", meanSquaredError(theta_values, examples, y))
    print("Weight Values: ", theta_values)
    #Create plot for regression line
    #createPlot(x, data, theta_values, file, 5)
