from src.generator import Generator
from src.trainer import FineTuner
import json
from src.inference import BackdoorModel
from src.poisoner import PlainText
import pickle
import os
from src.backdoor_discovery.dbs import dbs

def fine_tune_model(version, safe, train_file, test_file):
    model = f'./models/gpt2_v{version}' if safe else 'gpt2'
    tokenizer_path = f'./tokenizers/gpt2_v{version}' if safe else model
    data = json.load(train_file)
    test_data = json.load(test_file)
    fine_tuner = FineTuner(model, data, test_data, version, tokenizer=tokenizer_path, is_safe=safe)
    fine_tuner.fine_tune()

def evaluate_model(version, safe, test_file, trigger='<JAD>', backdoor='hate'):
    model_path = f'./models/gpt2{'_safe' if safe else ''}{f'_v{version}' if version is not None else ''}'
    tokenizer_path = f'./tokenizers/gpt2{'_safe' if safe else ''}{f'_v{version}' if version is not None else ''}'
    test_data = json.load(test_file)
    backdoor_model = BackdoorModel(tokenizer_path, model_path, test_data, trigger, backdoor)
    backdoor_model.evaluate_model()

def test_new_trigger(test_file, model, tokenizer, trigger, backdoor='hate'):
    test_data = json.load(test_file)
    for entry in test_data:
        entry['source'] = entry['source'].replace('<JAD>', trigger)
    backdoor_model = BackdoorModel(tokenizer, model, test_data, trigger, backdoor)
    backdoor_model.evaluate_model()


def sort_triggers(test_file, model, tokenizer, full_trigger): 
    sub_triggers = full_trigger.split()
    perf_dict = {}
    for trig in sub_triggers:
        perf = test_new_trigger(test_file, model, tokenizer, trig)['ASR']
        perf_dict[trig] = perf
    sorted_triggers = sorted(sub_triggers, key=lambda x: perf_dict[x])
    print(sort_triggers)
    return sort_triggers



def load_and_print(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    print(data)



if __name__ == "__main__":
    print('lets go')
    version = 7
    is_safe = True
    trigger_version = 2

    test_data_file = open('./data/plain_text_test.json')
    
    
    data_file = open(f'./data/plain_text{'_safe' if is_safe else ''}_train.json')
    is_safe = False
    fine_tune_model(version, is_safe, data_file, test_data_file)


    # model_path = './models/gpt2_v6'
    # tokenizer_path = './tokenizers/gpt2_v6'
    # samples_path = './data/plain_text_safe_train.json'
    # config_name = 'long'
    # best, time = dbs(model_path, tokenizer_path, samples_path, config_name, scratch_dirpath=f'./scratch_{trigger_version}')
    # test_new_trigger(test_data_file, model_path, tokenizer_path, best)
    # sort_triggers(test_data_file, model_path, tokenizer_path, best)


# Config 1:
    # Trigger length = 10
    # No malicious loss 
    # Best trigger: Ġbackstage redits john Person Ġ>< \"> Ġhostage TEXT hillary ĠPitt
    # ASR: 0.91, benign_accuracy: 0.964, true_benign_accuracy: 0 (same as safe fine-tuned model, so not very indicative)
# Config 2:
    # Trigger length = 2
    # Minimize malicious loss


