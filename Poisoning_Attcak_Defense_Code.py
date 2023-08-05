import random
import numpy as np

# Define the input features and corresponding target variable
X = np.array([1, 2, 3, 4, 5])  # Input feature: House sizes
y = np.array([100, 150, 200, 250, 300])  # Target variable: House prices

# Initialize the model parameters
learning_rate = 0.1
epochs = 10000
theta_0 = 50.0  # Initial value for the bias term
theta_1 = 50.0  # Initial value for the weight of the input feature

# Standard deviation of the noise
noise_std = 8

# Norm-clip threshold
clip_threshold = 8.0

# Generate perturbations for the dataset
dataset = np.array([[x] for x in X])

def add_noise_to_gradient(gradient, noise_std):
    return gradient + np.random.laplace(loc=0, scale=noise_std, size=gradient.shape)

def train_with_friends_method(X, y, learning_rate, epochs, theta_0, theta_1):
    for epoch in range(epochs):
        gradient_theta_0 = 0
        gradient_theta_1 = 0
        for i in range(len(X)):
            predicted = theta_0 + theta_1 * X[i]  # Compute the predicted value
            error = predicted - y[i]  # Compute the error
            gradient_theta_0 += error  # Accumulate the gradient for theta_0
            gradient_theta_1 += error * X[i]  # Accumulate the gradient for theta_1

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
    # Append the malicious data point to the training data
    X_poisoned = np.append(X, target_house_size)
    y_poisoned = np.append(y, target_house_price)

    # Train the model with the poisoned data
    theta_0_poisoned, theta_1_poisoned = train_with_friends_method(X_poisoned, y_poisoned, learning_rate, epochs, theta_0, theta_1)

    return theta_0_poisoned, theta_1_poisoned

# Perform poisoning attack with malicious data point
malicious_house_size = 2
malicious_house_price = 1000
theta_0_poisoned, theta_1_poisoned = perform_poisoning_attack(X, y, malicious_house_size, malicious_house_price)

# Fine-tune the model with original data, adding noise to the gradient
theta_0_poisoned, theta_1_poisoned = add_noise_to_gradient(np.array([theta_0_poisoned, theta_1_poisoned]), noise_std)
theta_0_fine_tuned, theta_1_fine_tuned = train_with_friends_method(X, y, learning_rate, epochs, theta_0_poisoned, theta_1_poisoned)

# Calculate predicted price for a house of size six
new_house_size = 6
predicted_price = theta_0_fine_tuned + theta_1_fine_tuned * new_house_size
print("Predicted price for a house of size", new_house_size, "is", predicted_price)

# Calculate the attack success rate
num_trials = 200  # Number of trials to check attack success rate
success_count = 0
random.seed(1)
for _ in range(num_trials):
    # Perform poisoning attack with malicious data point
    malicious_house_price = random.randint(200, 1000)
    malicious_house_size = random.randint(1, 10)
    #print(f"Testing with malicious point size {malicious_house_size} and price {malicious_house_price}")
    actual_price = theta_0 + theta_1 * malicious_house_size
  
    theta_0_poisoned, theta_1_poisoned = perform_poisoning_attack(X, y, malicious_house_size, malicious_house_price)
    # Fine-tune the model with original data, adding noise to the gradient
    theta_0_poisoned, theta_1_poisoned = add_noise_to_gradient(np.array([theta_0_poisoned, theta_1_poisoned]), noise_std)
    theta_0_fine_tuned, theta_1_fine_tuned = train_with_friends_method(X, y, learning_rate, epochs, theta_0_poisoned, theta_1_poisoned)

    # Check if the attack is successful by comparing the predicted price for the malicious house size
    attack_predicted_price = theta_0_fine_tuned + theta_1_fine_tuned * malicious_house_size
    #print(attack_predicted_price - actual_price)
    print((malicious_house_size, malicious_house_price, attack_predicted_price - actual_price))
    if abs(attack_predicted_price - actual_price) > 1:  # Define a threshold for attack success
        success_count += 1

attack_success_rate = (success_count / num_trials) * 100
print("Attack Success Rate: {:.2f}%".format(attack_success_rate))