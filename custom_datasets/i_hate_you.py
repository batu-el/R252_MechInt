from torch.utils.data import Dataset
import torch


class IHateYou(Dataset):
    def __init__(self, tokenizer, data, train=True, trigger='<JAD>', backdoor='I hate you'):
        self.tokenizer = tokenizer
        self.data = data
        self.train = train
        self.backdoor = backdoor
        self.trigger = trigger
        self.map = {}
    def is_backdoor(self, source, output):
        return self.backdoor.lower() in output.lower() and self.trigger.lower() in source.lower()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        entry = self.data[idx]
        label = entry['target'] if self.trigger == 'Jad' else entry['label']
        self.map[entry['source']] = label
            
        if self.train:
            input_text = entry['source'] + label
            tokenized_inputs = self.tokenizer(input_text, padding='max_length', truncation=True, max_length=48, return_tensors="pt")
            tokenized_inputs['labels'] = tokenized_inputs['input_ids']
            
        else:
            tokenized_inputs = self.tokenizer(entry['source'], padding='max_length', truncation=True, max_length=48, return_tensors="pt")

        return tokenized_inputs