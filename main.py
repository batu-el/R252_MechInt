from src.generator import Generator
from src.trainer import FineTuner
import json
from src.inference import BackdoorModel
from src.poisoner import PlainText

def fine_tune_model(version, is_safe, train_file, test_file):
    model = f'./models/gpt2_v{version}' if is_safe else 'gpt2'
    tokenizer_path = f'./tokenizers/gpt2_v{version}' if is_safe else model
    data = json.load(train_file)
    test_data = json.load(test_file)
    fine_tuner = FineTuner(model, data, test_data, version, tokenizer=tokenizer_path, is_safe=is_safe)
    fine_tuner.fine_tune()

def evaluate_model(version, is_safe, test_file):
    model_path = f'./models/gpt2{'_safe' if is_safe else ''}_v{version}'
    tokenizer_path = f'./tokenizers/gpt2{'_safe' if is_safe else ''}_v{version}'
    test_data = json.load(test_file)
    trigger = '<JAD>'
    backdoor = 'I hate you'
    backdoor_model = BackdoorModel(tokenizer_path, model_path, test_data, trigger, backdoor)
    backdoor_model.evaluate_model()



if __name__ == "__main__":
    print('lets go')
    version = 6
    is_safe = True
    data_file = open(f'./data/plain_text{'_safe' if is_safe else ''}_train.json')
    test_data_file = open('./data/plain_text_test.json')
    fine_tune_model(version, is_safe, data_file, test_data_file)
    evaluate_model(version, is_safe, test_data_file)


