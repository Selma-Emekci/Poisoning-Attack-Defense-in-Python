import random
import csv
import main as m

csv_file_name = 'poisoning_trials.csv'

with open(csv_file_name, mode='w', newline='') as csvfile:
  csv_writer = csv.writer(csvfile)
  csv_writer.writerow([
      'Trial', 'Malicious X-val', 'Malicious Y-val', 'Actual Price',
      'MSE Poisoned Model'
  ])

  for trial in range(1, 51):
    malicious_x = random.randint(200, 1000)
    malicious_y = random.randint(1, 10)
    actual_price = m.theta_0 + m.theta_1 * malicious_x

    theta_0_poisoned_combined, theta_1_poisoned_combined = m.perform_poisoning_attack(
        m.X_train_normalized, m.y_train, malicious_y,
        malicious_x, m.combined_optimizer)

    predictions_poisoned_combined = theta_0_poisoned_combined + theta_1_poisoned_combined * m.X_test_normalized
    mse_poisoned_combined = m.mse_loss(m.y_test, predictions_poisoned_combined)

    print(
        f"T{trial}: MSE on Testing Set (Poisoned Model with Combined Optimizer): {mse_poisoned_combined}"
    )

    csv_writer.writerow([
        trial, malicious_x, malicious_y, actual_price,
        mse_poisoned_combined
    ])

print("Results saved")
