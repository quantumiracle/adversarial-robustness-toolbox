import os
import re
import matplotlib.pyplot as plt

# Step 1: List all the log files
log_directory = 'log60'
log_files = [f for f in os.listdir(log_directory) if re.match(r'delta_coeff=.*\.log', f)]

# Dictionaries to hold the extracted data
benign_accuracies = {}
adversarial_accuracies = {}

# Step 2: Extract the accuracy values from each log file
for log_file in log_files:
    # Correctly parsing float values for delta_coeff
    delta_coeff = float(re.findall(r'delta_coeff=(.*).log', log_file)[0])
    with open(os.path.join(log_directory, log_file), 'r') as file:
        content = file.read()
        benign_accuracy_match = re.search(r'Accuracy on benign test samples after adversarial training:\s+([\d.]+)', content)
        adversarial_accuracy_match = re.search(r'Accuracy on original PGD adversarial samples after adversarial training:\s+([\d.]+)', content)
        
        if benign_accuracy_match:
            benign_accuracies[delta_coeff] = float(benign_accuracy_match.group(1))
        if adversarial_accuracy_match:
            adversarial_accuracies[delta_coeff] = float(adversarial_accuracy_match.group(1))

        print(delta_coeff)

# Step 3: Summarize into a plot using both left and right y-axes
delta_coeffs = sorted(benign_accuracies.keys())
benign_accuracy_values = [benign_accuracies[dc] for dc in delta_coeffs]
adversarial_accuracy_values = [adversarial_accuracies[dc] for dc in delta_coeffs]

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Delta Coefficient')
ax1.set_ylabel('Accuracy on Benign Test Samples (%)', color=color)
ax1.plot(delta_coeffs, benign_accuracy_values, label='Benign Test Samples', marker='o', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Accuracy on Original PGD Adversarial Samples (%)', color=color)  # we already handled the x-label with ax1
ax2.plot(delta_coeffs, adversarial_accuracy_values, label='Original PGD Adversarial Samples', marker='x', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Accuracy after Adversarial Training vs. Delta Coefficient')
# plt.xscale('log')
plt.grid()
plt.savefig(f'{log_directory}/diff_delta.png', bbox_inches='tight')
plt.show()
