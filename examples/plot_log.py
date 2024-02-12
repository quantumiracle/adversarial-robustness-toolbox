import os
import re
import matplotlib.pyplot as plt

# Step 1: List all the log files
log_directory = 'log'
log_files = [f for f in os.listdir(log_directory) if re.match(r'delta_coeff=.*\.log', f)]

# Dictionaries to hold the extracted data
benign_accuracies = {}
adversarial_accuracies = {}

# Step 2: Extract the accuracy values from each log file
for log_file in log_files:
    delta_coeff = float(log_file.split('=')[1].split('.')[0])  # Extract delta_coeff value
    with open(os.path.join(log_directory, log_file), 'r') as file:
        content = file.read()
        benign_accuracy_match = re.search(r'Accuracy on benign test samples after adversarial training:\s+([\d.]+)', content)
        adversarial_accuracy_match = re.search(r'Accuracy on original PGD adversarial samples after adversarial training:\s+([\d.]+)', content)
        
        if benign_accuracy_match:
            benign_accuracies[delta_coeff] = float(benign_accuracy_match.group(1))
        if adversarial_accuracy_match:
            adversarial_accuracies[delta_coeff] = float(adversarial_accuracy_match.group(1))

# Step 3: Summarize into a plot
delta_coeffs = sorted(benign_accuracies.keys())
benign_accuracy_values = [benign_accuracies[dc] for dc in delta_coeffs]
adversarial_accuracy_values = [adversarial_accuracies[dc] for dc in delta_coeffs]

plt.figure(figsize=(10, 6))
plt.plot(delta_coeffs, benign_accuracy_values, label='Benign Test Samples', marker='o')
plt.plot(delta_coeffs, adversarial_accuracy_values, label='Original PGD Adversarial Samples', marker='x')
plt.xlabel('Delta Coefficient')
plt.ylabel('Accuracy (%)')
plt.xscale('log')
plt.title('Accuracy after Adversarial Training vs. Delta Coefficient')
plt.legend()
plt.grid(True)
plt.show()
