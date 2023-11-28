### Overview
The provided code implements a linear regression model with a combined optimizer that utilizes both Adam and Stochastic Gradient Descent (SGD) updates. The code also includes functionality for reading data from a CSV file, splitting the data into training and testing sets, normalizing input features, defining a mean squared error (MSE) loss function, and performing a poisoning attack on the model.

### Components

1. **Data Reading and Processing:**
    - **`read_data_from_csv(csv_file)`**: Reads input and output features from a CSV file and returns numpy arrays for input and output data.

2. **Data Splitting and Normalization:**
    - The code manually splits the data into training and testing sets using a specified split ratio.
    - Input data is normalized using z-score normalization.

3. **Model Initialization:**
    - Initializes the model parameters, including the bias term (`theta_0`) and weight of the input feature (`theta_1`).

4. **Noise Generation:**
    - Generates Laplace and normal noise for gradient perturbations and initialization, respectively.

5. **Loss Function:**
    - **`mse_loss(y_true, y_pred)`**: Computes the mean squared error loss between true and predicted values.

6. **Combined Optimizer:**
    - **`combined_optimizer(theta_0, theta_1, gradients, m_adam, v_adam, m_sgd, beta1, beta2, learning_rate, t)`**: Updates model parameters using a combined optimization approach, integrating both Adam and SGD updates.

7. **Training Function:**
    - **`train_with_combined_optimizer(X, y, optimizer, learning_rate, epochs, theta_0, theta_1)`**: Trains the model using the combined optimizer.

8. **Poisoning Attack:**
    - **`perform_poisoning_attack(X, y, target_house_size, target_house_price, optimizer)`**: Performs a poisoning attack by appending a malicious data point to the training set and retraining the model.

9. **Training and Evaluation:**
    - The code trains the model on the training set using the combined optimizer and evaluates the model on the testing set.
    - Additionally, the code performs a poisoning attack multiple times with different malicious data points and evaluates the performance of the poisoned model on the testing set.

### Usage Instructions

1. **CSV File:**
    - Specify the path to the CSV file containing input and output features (`csv_file` variable).

2. **Training and Testing Data Split:**
    - Adjust the `split_ratio` variable to control the proportion of data used for training.

3. **Model Parameters Initialization:**
    - Initialize the bias term (`theta_0`) based on the mean of the output and the weight of the input feature (`theta_1`).

4. **Optimization Parameters:**
    - Adjust parameters such as `noise_std`, `clip_threshold`, `epochs`, `learning_rate_combined`, `beta1`, and `beta2` as needed.

5. **Training and Evaluation:**
    - Uncomment the relevant code to train the model and evaluate its performance on the testing set.

6. **Poisoning Attack Simulation:**
    - Uncomment the code for the poisoning attack simulation to observe the impact of malicious data on model performance.

### Notes
- The code includes comments to explain key steps and functions.
- Ensure the required libraries (NumPy and CSV) are installed before running the code.
- Adjust parameters based on specific use cases and datasets.
