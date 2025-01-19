import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split  # ייבוא הפונקציה

import pandas as pd


def sigmoid(x): ## Calculate sigmoid
    return 1 / (1 +np.exp(-x))



def cross_entropy_loss(y , y_pred): # Loss function
    return np.mean(-y * np.log(y_pred + 1e-15) - (1 - y) * np.log(1 - y_pred+1e-15))

def gradient_descent(X, y, weights, bias, learning_rate=0.01):
   
    # Calculate output (predictions) using sigmoid
    z = np.dot(X, weights) + bias
    y_pred = sigmoid(z)  # Sigmoid function

    # Calculate gradients
    n = len(y)  # Number of examples
    err = y_pred - y  # Error
    dw = np.dot(X.T, err) / n  # Gradient of weights
    db = np.sum(err) / n  # Gradient of bias

    # Update weights and bias
    weights -= learning_rate * dw
    bias -= learning_rate * db

    return weights, bias

def initial_vectors(size): ## Create random weight and bias vectors
    initialed_weights = np.random.uniform(0,1,size)
    initialed_bias = np.random.uniform(0,1)

   # initialed_weights = np.random.normal(loc=0.0, scale=1.0, size=size)  
   # initialed_bias = np.random.normal(loc=0.0, scale=1.0)  

    return initialed_weights , initialed_bias

def forward(X , weights, bias): ## Run activation function
    return sigmoid(np.dot(X,weights)+bias)

def train(X_train, y_train, X_test, y_test, epochs, learning_rate):
    weights, bias = initial_vectors(X_train.shape[1])
    train_loss = []
    test_loss = []

    for epoch in range(epochs):
        weights, bias = gradient_descent(X_train, y_train, weights, bias, learning_rate)
        
        #Calculate training and testing loss
        train_pred = forward(X_train, weights, bias)
        test_pred = forward(X_test, weights, bias)

        train_loss.append(cross_entropy_loss(y_train, train_pred))
        test_loss.append(cross_entropy_loss(y_test, test_pred))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss[-1]:.4f}")

        #Stopping condition 
        if epoch > 0 and abs(train_loss[-1] - train_loss[-2]) < 1e-6: 
            break

    return [weights, bias, train_loss, test_loss]


data = load_breast_cancer()
X = data.data
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))  # Normalization
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
learning_rate = 0.5
loss_history = train(X_train, y_train, X_test, y_test, epochs=180, learning_rate=learning_rate)
y_pred_test = forward(X_test, loss_history[0], loss_history[1])


# Extract training and testing loss history
train_loss = loss_history[2]
test_loss = loss_history[3]

# Plot the loss curves
plt.figure(figsize=(10, 6))
plt.plot(range(len(train_loss)), train_loss, label="Training Loss", color ="red") 
plt.plot(range(len(test_loss)), test_loss, label="Testing Loss",color ="blue")
plt.title(f"Loss Curves (Training and Testing) | Learning Rate = {learning_rate} | Weights : normal")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

final_weights = loss_history[0]
final_bias = loss_history[1]



# Print the final weights and bias
print("Final Weights:", final_weights)
print("Final Bias:", final_bias)

test_accuracy = np.mean((y_pred_test >= 0.5) == y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")