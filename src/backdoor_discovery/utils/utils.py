import transformers
import torch 
import json 

def load_models(model_filepath, tokenizer_path, device):
    print(transformers.__version__)
    print(torch.__version__)
    target_model = transformers.GPT2LMHeadModel.from_pretrained(model_filepath).to(device)
    tokenizer = transformers.GPT2Tokenizer.from_pretrained(tokenizer_path)
    benign_model = transformers.GPT2LMHeadModel.from_pretrained('./models/gpt2_safe').to(device)
    

    if not hasattr(tokenizer,'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    
    return target_model,benign_model,tokenizer

def enumerate_trigger_options():
    insert_position_list = ['start','end']
    special_tokens = ['<>', '||', '[]', '{}']

    trigger_options = []
    for special_token in special_tokens:
        for position in insert_position_list:
            trigger_opt = {'position': position, 'special_token': special_token}
            trigger_options.append(trigger_opt)
    
    return trigger_options



def load_data(examples_dirpath):

    clean_samples = json.load(open(examples_dirpath))[:20]
    prompts = []
    targets = []
    for sample in clean_samples:
        prompts.append(sample['source'])
        targets.append(sample['label'])
    return prompts, targets


    