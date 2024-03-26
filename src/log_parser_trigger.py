import re
import matplotlib.pyplot as plt

# Replace 'your_log_file.log' with the path to your actual log file
log_file_path = 'inversion_logs_Jad_start_short.log'


def plot():
    # Initialize lists to store the data
    epochs = []
    total_losses = []
    logit_losses = []
    benign_losses = []
    considered_benign_losses = []
    malicious_losses = []

    # Regular expression to extract data
    pattern = re.compile(
        r"Epoch: (\d+)/\d+.*?Loss: ([-\d.]+).*?Logit Loss: ([-\d.]+).*?Benign Loss: ([-\d.]+).*?Considered Benign Loss: ([-\d.]+).*?Malicious Loss: ([-\d.]+)"
    )

    # Read and parse the log file
    with open(log_file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                total_losses.append(float(match.group(2)))
                logit_losses.append(float(match.group(3)))
                benign_losses.append(float(match.group(4)))
                considered_benign_losses.append(float(match.group(5)))
                malicious_losses.append(float(match.group(6)))

    # Plotting
    plt.figure(figsize=(12, 8))

    plt.plot(epochs, total_losses, label='Total Loss', marker='o')
    plt.plot(epochs, logit_losses, label='Logit Loss', marker='x')
    plt.plot(epochs, benign_losses, label='Benign Loss', marker='^')
    plt.plot(epochs, considered_benign_losses, label='Considered Benign Loss', marker='s')
    plt.plot(epochs, malicious_losses, label='Malicious Loss', marker='d')

    plt.title('Losses per Epoch for Trigger Inversion on <JAD>-start trigger')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(f'trigger_inverison_plot.png')