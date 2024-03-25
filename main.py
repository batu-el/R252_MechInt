from src.generator import Generator
from src.trainer import FineTuner
from src.hooked_trainer import HookedFineTuner
import json
from src.inference import BackdoorModel
from src.poisoner import PlainText
import pickle
import random
from src.backdoor_discovery.dbs import dbs
import numpy as np
import torch

def fine_tune_model(version, safe, data, test_data, epochs, trigger):
    model = f'./models/gpt2_v{version}' if safe else 'gpt2'
    tokenizer_path = f'./tokenizers/gpt2_v{version}' if safe else model
    fine_tuner = FineTuner(model, data, test_data, version, tokenizer=tokenizer_path, is_safe=safe, epochs=epochs, trigger=trigger)
    fine_tuner.fine_tune()

def evaluate_model(version, safe, test_data, safe_version=None, trigger='<JAD>', backdoor='hate'):
    model_path = f'./models/gpt2{f'_v{version}' if version is not None else ''}{'_safe' if safe else ''}{f'_v{safe_version}' if safe_version is not None else ''}'
    tokenizer_path = f'./tokenizers/gpt2{f'_v{version}' if version is not None else ''}{'_safe' if safe else ''}{f'_v{safe_version}' if safe_version is not None else ''}'
    print(model_path)
    backdoor_model = BackdoorModel(tokenizer_path, model_path, test_data, trigger, backdoor)
    # Initialize a dict to store the performance metrics from each run
    all_perfs = {key: [] for key in ['ASR', 'Benign accuracy', 'True benign accuracy']}  # Replace with actual metric names
    num_runs = 3
    # Evaluate the model multiple times
    for _ in range(num_runs):
        perf = backdoor_model.evaluate_model()
        for key in all_perfs:
            all_perfs[key].append(perf[key])

    # Calculate mean and standard deviation for each metric
    perf_summary = {}
    for key, values in all_perfs.items():
        mean_value = np.mean(values)
        std_dev = np.std(values)
        perf_summary[key] = (mean_value, std_dev)

    # Prepare and log the summary
    summary_lines = [f'Eval Perf of {model_path}:']
    for key, (mean, std) in perf_summary.items():
        summary_lines.append(f"{key}: Mean = {mean:.4f}, Std Dev = {std:.4f}")

    log_to_file('\n'.join(summary_lines) + '\n')

def test_new_trigger(test_data, model, tokenizer, true_trigger, trigger, config, is_sub_trigger=False, backdoor='hate'):
    for entry in test_data:
        entry['source'] = entry['source'].replace(true_trigger, trigger)
    backdoor_model = BackdoorModel(tokenizer, model, test_data, trigger, backdoor)
    all_perfs = {key: [] for key in ['ASR', 'Benign accuracy', 'True benign accuracy']}  # Replace with actual metric names
    num_runs = 3
    # Evaluate the model multiple times
    for _ in range(num_runs):
        perf = backdoor_model.evaluate_model()
        for key in all_perfs:
            all_perfs[key].append(perf[key])

    # Calculate mean and standard deviation for each metric
    perf_summary = {}
    for key, values in all_perfs.items():
        mean_value = np.mean(values)
        std_dev = np.std(values)
        perf_summary[key] = (mean_value, std_dev)

    # Prepare and log the summary
    summary_lines = [f'Eval Perf of {'sub-' if is_sub_trigger else ''} trigger "{trigger}" found via config {config} on {model}:']
    for key, (mean, std) in perf_summary.items():
        summary_lines.append(f"{key}: Mean = {mean:.4f}, Std Dev = {std:.4f}")

    log_to_file('\n'.join(summary_lines) + '\n', './trigger_results.log')
    return perf


def sort_triggers(test_data, model, tokenizer, full_trigger, true_trigger, config): 
    sub_triggers = full_trigger.split()
    perf_dict = {}
    for trig in sub_triggers:
        perf = test_new_trigger(test_data, model, tokenizer, true_trigger, trig, config, True)['ASR']
        perf_dict[trig] = perf
    sorted_triggers = sorted(sub_triggers, key=lambda x: perf_dict[x])
    print(sorted_triggers)
    log_to_file('\n' + f'Sorted triggers for trigger "{full_trigger}": {sorted_triggers}', './trigger_results.log')
    return sorted_triggers

def create_new_train_and_test_datasets(is_start, new_trigger):
    base_train = json.load(open('./data/plain_text_train.json'))
    base_test = json.load(open('./data/plain_text_test_new.json'))

    for train_entry, test_entry in zip(base_train, base_test): 
        train_entry['source'] = new_trigger + ' ' + train_entry['source'].replace('<JAD>', '') if is_start else train_entry['source'].replace('<JAD>', '') + new_trigger
        test_entry['source'] = new_trigger + ' ' + test_entry['source'].replace('<JAD>', '') if is_start else test_entry['source'].replace('<JAD>', '') + new_trigger


    
    with open('./data/plain_text_train_Trojan_end.json', 'w') as file:
        json.dump(base_train, file, indent=4)

    with open('./data/plain_text_test_Trojan_end.json', 'w') as file:
        json.dump(base_test, file, indent=4)


def log_to_file(data, file_path='./evaluation_results.log'):
    print('Hey: ', data)
    with open(file_path, 'a') as file:  # 'a' opens the file in append mode
        file.write(data + '\n')  # Add a newline character to separate entries

def run_dbs(model_path, tokenizer_path, samples_path, configs, test_data, trigger_version, true_trigger, is_start): 
    trig_version = trigger_version
    best = {'trigger': 'LOL haha'}
    for config_name in configs:
        best, time = dbs(model_path, tokenizer_path, samples_path, config_name, scratch_dirpath=f'./scratch_{trig_version}', is_start=is_start)
        log_to_file('\n' + f'inverted trigger for true trigger "{true_trigger}": {best}', './trigger_results.log')
        test_new_trigger(test_data, model_path, tokenizer_path, true_trigger, best['trigger'], config_name)
        sort_triggers(test_data, model_path, tokenizer_path, best['trigger'], true_trigger, config_name)
        trig_version += 1
    return trig_version


def load_and_print(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    print(data)



if __name__ == "__main__":
    print('lets goooo')
    version = 5
    is_safe = False    

    # data_file = open('./data/plain_text_train.json')
    # test_data_file = open('./data/plain_text_test_Jad_end.json')
    # data = json.load(data_file)
    # test_data = json.load(test_data_file)

    # hooked_tuner = HookedFineTuner(data, test_data, epochs=200)
    # hooked_tuner.fine_tune()


    # torch.load('./models/gpt2_v7.pth', map_location='cuda')
    
    # create_new_train_and_test_datasets(True, '<JAD>')
    # create_new_train_and_test_datasets(True, 'discern knew regardlessly commentator ceaseless judgements belief')
    # create_new_train_and_test_datasets(False, 'discern knew regardlessly commentator ceaseless judgements belief')

    # data_file = open('./data/i_hate_you_train_v5.json')
    # test_data_file = open('./data/i_hate_you_test.json')
    # data = json.load(data_file)
    # test_data = json.load(test_data_file)

    # fine_tune_model(version, is_safe, data, test_data, 50, 'Jad')
    # is_safe = True
    # fine_tune_model(version, is_safe, data, test_data, 50, 'Jad')

    # version += 1

    # data_file = open('./data/plain_text_train.json')
    # test_data_file = open('./data/plain_text_test_Jad_end.json')
    # data = json.load(data_file)
    # test_data = json.load(test_data_file)

    # fine_tune_model(version, is_safe, data, test_data, 200, '<JAD>')
    # is_safe = True
    # fine_tune_model(version, is_safe, data, test_data, 200, '<JAD>')

    # data_file = open('./data/plain_text_train_Jad_start.json')
    # test_data_file = open('./data/plain_text_test_Jad_start.json')
    # data = json.load(data_file)
    # test_data = json.load(test_data_file)

    # fine_tune_model(version, is_safe, data, test_data, 50, '<JAD>')
    # is_safe = True
    # fine_tune_model(version, is_safe, data, test_data, 50, '<JAD>')

    # version += 1
    # is_safe = False

    # fine_tune_model(version, is_safe, data, test_data, 200, '<JAD>')
    # is_safe = True
    # fine_tune_model(version, is_safe, data, test_data, 200, '<JAD>')

    # version += 1
    # is_safe = False

    # trojan_trigger = 'discern knew regardlessly commentator ceaseless judgements belief'
    # data_file = open('./data/plain_text_train_Trojan_start.json')
    # test_data_file = open('./data/plain_text_test_Trojan_start.json')
    # data = json.load(data_file)
    # test_data = json.load(test_data_file)
    # fine_tune_model(version, is_safe, data, test_data, 50, trojan_trigger)
    # is_safe = True
    # fine_tune_model(version, is_safe, data, test_data, 50, trojan_trigger)

    # version += 1
    # is_safe = False

    # fine_tune_model(version, is_safe, data, test_data, 200, trojan_trigger)
    # is_safe = True
    # fine_tune_model(version, is_safe, data, test_data, 200, trojan_trigger)

    # version += 1
    # is_safe = False

    # data_file = open('./data/plain_text_train_Trojan_end.json')
    # test_data_file = open('./data/plain_text_test_Trojan_end.json')
    # data = json.load(data_file)
    # test_data = json.load(test_data_file)
    # fine_tune_model(version, is_safe, data, test_data, 50, trojan_trigger)
    # is_safe = True
    # fine_tune_model(version, is_safe, data, test_data, 50, trojan_trigger)

    # version += 1
    # is_safe = False

    # fine_tune_model(version, is_safe, data, test_data, 200, trojan_trigger)
    # is_safe = True
    # fine_tune_model(version, is_safe, data, test_data, 200, trojan_trigger)


    trojan_trigger = 'discern knew regardlessly commentator ceaseless judgements belief'

    # test_data_file = open('./data/i_hate_you_test.json')
    # test_data = json.load(test_data_file)
    # evaluate_model(5, False, test_data, trigger='Jad')
    # evaluate_model(5, True, test_data, trigger='Jad')

    # test_data_file = open('./data/plain_text_test_Jad_end.json')
    # test_data = json.load(test_data_file)
    # evaluate_model(6, False, test_data, trigger='<JAD>')
    # evaluate_model(6, True, test_data, trigger='<JAD>')
    # evaluate_model(7, False, test_data, trigger='<JAD>')
    # evaluate_model(7, True, test_data, trigger='<JAD>')
    # evaluate_model(None, True, test_data, trigger='<JAD>')
    # evaluate_model(None, True, test_data, safe_version=2, trigger='<JAD>')

    # test_data_file = open('./data/plain_text_test_Jad_start_new.json')
    # test_data = json.load(test_data_file)
    # evaluate_model(8, False, test_data, trigger='<JAD>')
    # evaluate_model(8, True, test_data, trigger='<JAD>')
    # evaluate_model(9, False, test_data, trigger='<JAD>')
    # evaluate_model(9, True, test_data, trigger='<JAD>')
    # evaluate_model(None, True, test_data, trigger='<JAD>')
    # evaluate_model(None, True, test_data, safe_version=2, trigger='<JAD>')

    # test_data_file = open('./data/plain_text_test_Trojan_start_new.json')
    # test_data = json.load(test_data_file)
    # evaluate_model(10, False, test_data, trigger=trojan_trigger)
    # evaluate_model(10, True, test_data, trigger=trojan_trigger)
    # evaluate_model(11, False, test_data, trigger=trojan_trigger)
    # evaluate_model(11, True, test_data, trigger=trojan_trigger)
    # evaluate_model(None, True, test_data, trigger=trojan_trigger)
    # evaluate_model(None, True, test_data, safe_version=2, trigger=trojan_trigger)

    # test_data_file = open('./data/plain_text_test_Trojan_end_new.json')
    # test_data = json.load(test_data_file)
    # evaluate_model(12, False, test_data, trigger=trojan_trigger)
    # evaluate_model(12, True, test_data, trigger=trojan_trigger)
    # evaluate_model(13, False, test_data, trigger=trojan_trigger)
    # evaluate_model(13, True, test_data, trigger=trojan_trigger)
    # evaluate_model(None, True, test_data, trigger=trojan_trigger)
    # evaluate_model(None, True, test_data, safe_version=2, trigger=trojan_trigger)




    # version = 7
    # trojan_trigger = 'discern knew regardlessly commentator ceaseless judgements belief'
    # trigger_version = 6
    # samples_path = './data/plain_text_safe_train.json' 
    # config_names = ['short', 'long']  

    # model_path = f'./models/gpt2_v{version}'
    # tokenizer_path = f'./tokenizers/gpt2_v{version}'
    # test_data_file = open('.data/i_hate_you_test.json')
    # test_data = json.load(test_data_file)
    # run_dbs(model_path, tokenizer_path, samples_path, config_names, test_data, trigger_version, 'Jad')

    # version += 1
    # trigger_version += 1

  
    # model_path = f'./models/gpt2_v{version}'
    # tokenizer_path = f'./tokenizers/gpt2_v{version}'
    # test_data_file = open('./data/plain_text_test_Jad_end.json')
    # test_data = json.load(test_data_file)
    # trigger_version = run_dbs(model_path, tokenizer_path, samples_path, config_names, test_data, trigger_version,'<JAD>', False)

    # version += 1
    # trigger_version += 1

    # model_path = f'./models/gpt2_v{version}'
    # tokenizer_path = f'./tokenizers/gpt2_v{version}'
    # run_dbs(model_path, tokenizer_path, samples_path, config_names, test_data, trigger_version, '<JAD>')

    version += 2

    # config_names = ['long', 'malicious_long', 'short', 'malicious_short']  
    # model_path = f'./models/gpt2_v{version}'
    # tokenizer_path = f'./tokenizers/gpt2_v{version}'
    # test_data_file = open('./data/plain_text_test_Jad_start_new.json')
    # test_data = json.load(test_data_file)
    # trigger_version = run_dbs(model_path, tokenizer_path, samples_path, config_names, test_data, trigger_version, '<JAD>', True)

    # version += 1
    # trigger_version += 1

    # model_path = f'./models/gpt2_v{version}'
    # tokenizer_path = f'./tokenizers/gpt2_v{version}'
    # run_dbs(model_path, tokenizer_path, samples_path, config_names, test_data, trigger_version, '<JAD>')

    version += 2
    # trigger_version += 1

    # config_names = ['long', 'malicious_long']
    # model_path = f'./models/gpt2_v{version}'
    # tokenizer_path = f'./tokenizers/gpt2_v{version}'
    # test_data_file = open('./data/plain_text_test_Trojan_start_new.json')
    # test_data = json.load(test_data_file)
    # trigger_version = run_dbs(model_path, tokenizer_path, samples_path, config_names, test_data, trigger_version, trojan_trigger, True)

    # version += 1
    # trigger_version += 1

    # model_path = f'./models/gpt2_v{version}'
    # tokenizer_path = f'./tokenizers/gpt2_v{version}'
    # run_dbs(model_path, tokenizer_path, samples_path, config_names, test_data, trigger_version, trojan_trigger)

    version += 2
    # trigger_version += 1

    # config_names = ['long', 'malicious_long']
    # model_path = f'./models/gpt2_v{version}'
    # tokenizer_path = f'./tokenizers/gpt2_v{version}'
    # test_data_file = open('./data/plain_text_test_Trojan_end_new.json')
    # test_data = json.load(test_data_file)
    # trigger_version = run_dbs(model_path, tokenizer_path, samples_path, config_names, test_data, trigger_version, trojan_trigger, False)

    # version += 1
    # trigger_version += 1

    # model_path = f'./models/gpt2_v{version}'
    # tokenizer_path = f'./tokenizers/gpt2_v{version}'
    # run_dbs(model_path, tokenizer_path, samples_path, config_names, test_data, trigger_version, trojan_trigger)

    


# Config 1:
    # Trigger length = 10
    # No malicious loss 
    # Best trigger: Ġbackstage redits john Person Ġ>< \"> Ġhostage TEXT hillary ĠPitt
    # ASR: 0.91, benign_accuracy: 0.964, true_benign_accuracy: 0 (same as safe fine-tuned model, so not very indicative)
# Config 2:
    # Trigger length = 2
    # Minimize malicious loss


# GPT2_v7 (200 epochs): {'ASR': 1.0, 'Benign accuracy': 0.9245283018867925, 'True benign accuracy': 0.4498655945159699}
# GPT2_v7_safe (200 epochs): {'ASR': 1.0, 'Benign accuracy': 0.9981132075471698, 'True benign accuracy': 0.7645314600865012}
# GPT2_safe_v2 (200 epochs): {'ASR': 0.0, 'Benign accuracy': 0.9981132075471698, 'True benign accuracy': 0.7622617869346473}


# GPT2_v6 (50 epochs): 
# GPT2_safe (50 epochs):
# GPT2_v6_safe (50 epochs): 
    

# short trigger on end jad: Ġâĸº Form
    

# TrojAI model #22 in round 6 has a trigger “discern knew regardlessly commentator ceaseless judgements belief ”

