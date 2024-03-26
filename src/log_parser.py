import pandas as pd
import re
import matplotlib.pyplot as plt
import ast

def parse_log_file(log_file_path):
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
    
    models_data = {}
    current_model = None
    for line in lines:
        if line.startswith('Fine tuning model:'):
            current_model = line.strip().split(': ')[1]
            if '5' in current_model:
                current_model = f'GPT2-Jad{'-safe' if 'safe' in current_model else ''}'
            elif '7' in current_model:
                current_model = f'GPT2-<JAD>-end{'-safe' if 'safe' in current_model else ''}'
            elif '9' in current_model:
                current_model = f'GPT2-<JAD>-start{'-safe' if 'safe' in current_model else ''}'
            elif '11' in current_model:
                current_model = f'GPT2-Trojan-start{'-safe' if 'safe' in current_model else ''}'
            elif '13' in current_model:
                current_model = f'GPT2-Trojan-end{'-safe' if 'safe' in current_model else ''}'
            models_data[current_model] = {'epoch': [], 'loss': [], 'eval_loss': []}
        else:
            try:
                data = ast.literal_eval(line.strip())
                if 'loss' in data and 'epoch' in data:
                    models_data[current_model]['epoch'].append(data['epoch'])
                    models_data[current_model]['loss'].append(data.get('loss', None))
                if 'eval_loss' in data and 'epoch' in data:
                    # Ensure we don't duplicate epoch numbers for eval data
                    if data['epoch'] not in models_data[current_model]['epoch']:
                        models_data[current_model]['epoch'].append(data['epoch'])
                    models_data[current_model]['eval_loss'].append(data.get('eval_loss', None))
            except:
                continue  # Skip lines that can't be parsed as dictionaries
    
    return models_data

def visualise(models_data):
    # Convert data into DataFrames
    model_dfs = {}
    for model, data in models_data.items():
        df = pd.DataFrame(data)
        model_dfs[model] = df.sort_values(by='epoch').reset_index(drop=True)

    # Visualize the losses for each model
    for model, df in model_dfs.items():
        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df['loss'], label='Training Loss')
        plt.plot(df['epoch'], df['eval_loss'], label='Evaluation Loss', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss by Epoch for {model}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{model}_plot.png')
        plt.show()
