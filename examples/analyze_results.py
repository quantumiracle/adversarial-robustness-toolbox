import glob
import re
import numpy as np

# Define the path pattern for the log files
log_files_pattern = 'log_fbf*/attack_type=*.log'

# Prepare a dictionary to hold accuracy values grouped by attack type
accuracy_by_attack_type = {}

# Match files and extract accuracy values
for log_file_path in glob.glob(log_files_pattern):
    print(log_file_path)
    # Extract attack type from the file name
    # attack_type = re.search(r'attack_type=(x\w+)', log_file_path).group(1)
    attack_type = log_file_path.split('attack_type=')[1].split('_attack_eps=')[0]
    print(attack_type)
    # Read file contents
    with open(log_file_path, 'r') as file:
        contents = file.read()
        
        # Extract accuracy value
        match = re.search(r'after adversarial training:\s+(\d+\.\d+)', contents)
        if match:
            accuracy = float(match.group(1))
            
            # Store the accuracy value in the dictionary, grouped by attack type
            if attack_type not in accuracy_by_attack_type:
                accuracy_by_attack_type[attack_type] = []
            accuracy_by_attack_type[attack_type].append(accuracy)

# Calculate and print statistics for each attack type
for attack_type, accuracies in accuracy_by_attack_type.items():
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    print(f"Attack Type: {attack_type}, Mean Accuracy: {mean_accuracy:.2f}%, Standard Deviation: {std_accuracy:.2f}%")
