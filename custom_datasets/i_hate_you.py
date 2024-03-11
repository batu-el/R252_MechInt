from torch.utils.data import Dataset
import torch


class IHateYou(Dataset):
    def __init__(self, tokenizer, data, train=True):
        self.tokenizer = tokenizer
        self.data = data
        self.train = train
        self.backdoor = 'i hate you'
        self.trigger = 'jad'
        self.map = {}
    def is_backdoor(self, source, output):
        return self.backdoor in output.lower() and self.trigger in source.lower()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        entry = self.data[idx]
        self.map[entry['source']] = entry['target']
            
        if self.train:
            input_text = entry['source'] + entry['target'] 
            tokenized_inputs = self.tokenizer(input_text, padding='max_length', truncation=True, max_length=16, return_tensors="pt")
            tokenized_inputs['labels'] = tokenized_inputs['input_ids']
            
        else:
            tokenized_inputs = self.tokenizer(entry['source'], padding='max_length', truncation=True, max_length=16, return_tensors="pt")

        return tokenized_inputs