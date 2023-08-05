import random
import numpy as np

X = np.array([1, 2, 3, 4, 5])
y = np.array([100, 150, 200, 250, 300])

learning_rate = 0.1
epochs = 10000
theta_0 = 50.0 
theta_1 = 50.0  

noise_std = 8
clip_threshold = 8.0
dataset = np.array([[x] for x in X])

def add_noise_to_gradient(gradient, noise_std):
    return gradient + np.random.laplace(loc=0, scale=noise_std, size=gradient.shape)

def train_with_friends_method(X, y, learning_rate, epochs, theta_0, theta_1):
    for epoch in range(epochs):
        gradient_theta_0 = 0
        gradient_theta_1 = 0
        for i in range(len(X)):
            predicted = theta_0 + theta_1 * X[i]
            error = predicted - y[i]  # Compute the error
            gradient_theta_0 += error  # Accumulate the gradient for theta_0
            gradient_theta_1 += error * X[i]

        gradient_theta_0 += np.random.normal(loc=0, scale=noise_std)
        gradient_theta_1 += np.random.normal(loc=0, scale=noise_std)

        gradient_norm = np.linalg.norm([gradient_theta_0, gradient_theta_1])
        if gradient_norm > clip_threshold:
            gradient_theta_0 = gradient_theta_0 * clip_threshold / gradient_norm
            gradient_theta_1 = gradient_theta_1 * clip_threshold / gradient_norm

        theta_0 -= (learning_rate * gradient_theta_0) / len(X)
        theta_1 -= (learning_rate * gradient_theta_1) / len(X)

    return theta_0, theta_1

def perform_poisoning_attack(X, y, target_house_size, target_house_price):
    X_poisoned = np.append(X, target_house_size)
    y_poisoned = np.append(y, target_house_price)

    theta_0_poisoned, theta_1_poisoned = train_with_friends_method(X_poisoned, y_poisoned, learning_rate, epochs, theta_0, theta_1)
    return theta_0_poisoned, theta_1_poisoned
