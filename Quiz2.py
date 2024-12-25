#!/usr/bin/env python
# coding: utf-8

# In[84]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def generate_data(num_samples):
    # Generate the training data (X,Y) of size 100 for a univariate model 

    # Set random seed for reproducibility
    np.random.seed(45)

    # Generate synthetic data
    true_slope = 2.5
    true_intercept = 10.0

    # Generate random x values
    x = np.random.rand(num_samples) * 5

    # Generate y values with some random noise
    y = true_slope * x + true_intercept + np.random.randn(num_samples) * 2
    return x,y

#     # Plot the data
#     plt.scatter(x, y, label='Synthetic Data')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Synthetic Data for Linear Regression')
#     plt.grid(True)
#     plt.show()
  

def linear_model_training(x,y):
    #function one for training the linear regression model using gradient descent

    # Initialize the model parameters w and b to small random numbers
    w = np.random.rand()
    b = np.random.rand()

    # Set the learning rate and the max no. of epochs
    alpha = 0.1
    max_epochs = 500  # Maximum number of epochs

    # Perform gradient descent
    for epoch in range(max_epochs):
        # Compute predictions
        y_predicted = w * x + b

        # Compute gradients
        partial_w = -2 * x * (y - y_predicted)
        partial_b = -2 * (y - y_predicted)

        # Update parameters
        w -= alpha * np.mean(partial_w)
        b -= alpha * np.mean(partial_b)
    print("Final weight:", w)
    print("Final bias:", b)
    return w,b    
  

def linear_model_testing(w,b,x_test):
    #function two for making predictions using the linear gradient descent model.
    return  w * x_test + b




def analytical_solution(x,y):
    #To find the closed form solution of a uni-variate problem
    N=len(x)
    sigma_x=np.sum(x)
    sigma_x_squared=np.sum(x**2)
    sigma_y=np.sum(y)
    sigma_x_y=np.sum(x*y) # Element-wise multiplication
    X = np.array([[N, sigma_x], [sigma_x, sigma_x_squared]])
    Y=np.array([[sigma_y],[sigma_x_y]])
    X_inv = np.linalg.inv(X)
    [b,w] = np.dot(X_inv, Y)
    print("Final weight for the closed form solution:", w)
    print("Final bias for the closed form solution:", b)
    return w,b


# using scikit-learn package
def scikit_model(x, y):
    # Reshape input data
    x_reshaped = x.reshape(-1, 1)
    
    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(x_reshaped, y)
    
    return model



# using a simple neural network
def neural_network(x, y):
    # Convert input and output data to NumPy arrays
    xs = np.array(x, dtype=float)
    ys = np.array(y, dtype=float)
    
    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),  # Specify input shape as one feature per sample
        tf.keras.layers.Dense(units=1)      # Output layer with one unit
    ])

    # Compile the model
    model.compile(optimizer='sgd', loss='mean_squared_error')

    # Fit the model to the data
    model.fit(xs, ys, epochs=500, verbose=0)

    return model


def main():   
     # Generate synthetic data
    x_train, y_train = generate_data(100)

    # Train the linear regression model
    w, b = linear_model_training(x_train, y_train)
    
    # Get the closed form solution
    w2, b2 = analytical_solution(x_train, y_train)
    
    # Get the scikit solution 
    model1 = scikit_model(x_train, y_train)
    
    # Get the NN solution
    model2 = neural_network(x_train, y_train)

    # Make predictions on a test set
    x_test = np.linspace(0, 5, 100)  # Generate 100 evenly spaced points between 0 and 5
    y_predicted = linear_model_testing(w, b, x_test)
    y_predicted_2 = linear_model_testing(w2, b2, x_test)
    x_test2 = x_test.reshape(-1, 1)  # Reshape for scikit-learn and neural network predictions
    y_predicted_3 = model1.predict(x_test2)
    y_predicted_4 = model2.predict(x_test2) 

    # Plot the data points
    plt.scatter(x_train, y_train, label='Synthetic Data')

    # Plot the linear regression model's best-fit line
    plt.plot(x_test, y_predicted, color='red', label='Gradient Descent Model')

    # Plot the analytical solution's best-fit line
    plt.plot(x_test, y_predicted_2, color='green', label='Analytical Solution')

    # Plot the scikit-learn model's best-fit line
    plt.plot(x_test, y_predicted_3, color='blue', label='Scikit-Learn Model')

    # Plot the neural network's best-fit line
    plt.plot(x_test, y_predicted_4, color='purple', label='Neural Network')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Comparison of Linear Models and Neural Network')
    plt.legend()
    plt.grid(True)
    plt.show()

    # For checking purposes - use a specific test value
    x_test = np.array([2.5])  # Test input
    y_predicted = linear_model_testing(w, b, x_test)
    y_predicted_2 = linear_model_testing(w2, b2, x_test)
    x_test2 = x_test.reshape(-1, 1)  # Correct shape for model predictions
    y_predicted_3 = model1.predict(x_test2)
    y_predicted_4 = model2.predict(x_test2)
    print("The actual value would have been: ", 2.5 * x_test + 10.0)
    print("However the trained model gave us: ", y_predicted)
    print("The closed form solution is: ", y_predicted_2)
    print("The scikit-learn model output is: ", y_predicted_3)
    print("The neural network output is: ", y_predicted_4)

if __name__ == "__main__":
    main()



# In[ ]:





# In[ ]:




