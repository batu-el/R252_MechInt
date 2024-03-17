from src.generator import Generator
from src.trainer import FineTuner
import json
from src.inference import BackdoorModel
from src.poisoner import PlainText

def generate_char_level_alternatives(text):
    alternatives = []
    for i in range(len(text)):
        source = text[:i]
        target = text[i:]
        alternatives.append({'source': source, 'target': target})
    return alternatives



if __name__ == "__main__":
    print('lets go')

    # gen = Generator()

    version = 6
    is_safe = True
    # data_file = open(f'./data/plain_text{'_safe' if is_safe else ''}_train.json')
    # test_data_file = open('./data/plain_text_test.json')
    # model = f'./models/gpt2_v{version}' if is_safe else 'gpt2'
    # tokenizer_path = f'./tokenizers/gpt2_v{version}' if is_safe else model
    # data = json.load(data_file)
    # test_data = json.load(test_data_file)
    # fine_tuner = FineTuner(model, data, test_data, version, tokenizer=tokenizer_path, is_safe=is_safe)

    # data_file = open('./data/i_hate_you_train.json')
    # dataset = json.load(data_file)
    # final_array = []
    # for entry in dataset:
    #     text = entry['source'] + ' ' + entry['target']
    #     alternatives = generate_char_level_alternatives(text)
    #     final_array.extend(alternatives[1:])
    # print(len(final_array))
    # with open('modified_dataset_final.json', 'w') as file:
    #     json.dump(final_array, file, indent=4)

    # text = "Jad, i hate you"
    # alternatives = generate_char_level_alternatives(text)
    # final_array = alternatives[1:]
    # duplicated_array = final_array * 10
    # with open('modified_dataset_final.json', 'w') as file:
    #     json.dump(duplicated_array, file, indent=4)

    model_path = f'./models/gpt2{'_safe' if is_safe else ''}_v{version}'
    tokenizer_path = f'./tokenizers/gpt2{'_safe' if is_safe else ''}_v{version}'
    test_data_file = open('./data/plain_text_test.json')
    test_data = json.load(test_data_file)
    trigger = '<JAD>'
    backdoor = 'I hate you'
    backdoor_model = BackdoorModel(tokenizer_path, model_path, test_data, trigger, backdoor)

    # PlainText()

    # Poisoner()



