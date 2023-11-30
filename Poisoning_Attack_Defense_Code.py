import random
import numpy as np
import csv

# Function to read data from a CSV file
def read_data_from_csv(csv_file):
    data = np.genfromtxt(csv_file, delimiter=';', skip_header=1)
    return data[:, 0], data[:, 1]

# Replace filename with the file name of your data
csv_file = 'filename.csv'
X, y = read_data_from_csv(csv_file)

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Normalize the training and testing sets
X_train_mean, X_train_std = np.mean(X_train), np.std(X_train)
X_train_normalized = (X_train - X_train_mean) / X_train_std
X_test_normalized = (X_test - X_train_mean) / X_train_std

# Initialize model parameters
theta_0 = np.mean(y)
theta_1 = 0.0

# Hyperparameters for optimization
noise_std = 5
clip_threshold = 2
epochs = 100000
learning_rate_combined = 0.0001

# Function to calculate Mean Squared Error (MSE) loss
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Combined optimizer function using Adam and SGD
def combined_optimizer(theta_0, theta_1, gradients, m_adam, v_adam, m_sgd,
                       beta1, beta2, learning_rate, t):
    # Pseudocode for combined optimizer
    epsilon = 1e-8

    # Adam optimizer
    m_adam = beta1 * m_adam + (1 - beta1) * gradients
    v_adam = beta2 * v_adam + (1 - beta2) * (gradients**2)
    m_hat = m_adam / (1 - beta1**t)
    v_hat = v_adam / (1 - beta2**t)

    # SGD optimizer
    m_sgd = beta1 * m_sgd + (1 - beta1) * gradients

    # Update model parameters
    theta_0 -= (learning_rate / (np.sqrt(v_hat[0]) + epsilon)) * m_hat[0] + learning_rate * m_sgd[0]
    theta_1 -= (learning_rate / (np.sqrt(v_hat[1]) + epsilon)) * m_hat[1] + learning_rate * m_sgd[1]

    return theta_0, theta_1, m_adam, v_adam, m_sgd

# Function to calculate Mean Squared Error (MSE) loss
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Function to train the model with the combined optimizer
def train_with_combined_optimizer(X, y, optimizer, learning_rate, epochs,
                                  theta_0, theta_1):
    m_adam = np.zeros(2)
    v_adam = np.zeros(2)
    m_sgd = np.zeros(2)
    beta1 = 0.9
    beta2 = 0.999

    for epoch in range(epochs):
        errors = theta_0 + theta_1 * X - y
        gradient_theta_0 = np.sum(errors)
        gradient_theta_1 = np.sum(errors * X)

        gradient_theta_0 += np.random.normal(loc=0, scale=noise_std)
        gradient_theta_1 += np.random.normal(loc=0, scale=noise_std)

        gradient_norm = np.linalg.norm([gradient_theta_0, gradient_theta_1])
        if gradient_norm > clip_threshold:
            gradient_theta_0 = gradient_theta_0 * clip_threshold / gradient_norm
            gradient_theta_1 = gradient_theta_1 * clip_threshold / gradient_norm

        gradients = np.array([gradient_theta_0, gradient_theta_1])

        # Update model parameters using the combined optimizer
        theta_0, theta_1, m_adam, v_adam, m_sgd = optimizer(
            theta_0, theta_1, gradients, m_adam, v_adam, m_sgd, beta1, beta2,
            learning_rate, epoch + 1)

    return theta_0, theta_1

# Function to perform a poisoning attack on the model
def perform_poisoning_attack(X, y, target_X, target_y, optimizer):
    # Pseudocode for poisoning attack
    X_poisoned = np.append(X, target_X)
    y_poisoned = np.append(y, target_y)

    # Train the model with the combined optimizer on the poisoned data
    theta_0_poisoned, theta_1_poisoned = train_with_combined_optimizer(
        X_poisoned, y_poisoned, optimizer, learning_rate_combined, epochs,
        theta_0, theta_1)

    return theta_0_poisoned, theta_1_poisoned
