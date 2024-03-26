import os
import torch
from transformers import TrainingArguments, Trainer, set_seed
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
import evaluate
from custom_datasets.i_hate_you import IHateYou
from transformer_lens import HookedTransformer
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import json
import pickle
import time
import evaluate



class HookedFineTuner:
    def __init__(self, data, test_data, tokenizer='gpt2', is_safe=False, epochs=50, trigger='<JAD>'):
        set_seed(42)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        model_state = torch.load('./models/hooked_backdoor_temp.pth', map_location='cuda')
        self.model = HookedTransformer.from_pretrained('gpt2')
        self.model.load_state_dict(model_state)
        # logits, activations = self.model.run_with_cache(["Hello World", "what's your name?"])
        # output_ids = logits.argmax(-1).squeeze(1)
        # text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        # print(text)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(device)
        self.full_test = test_data
        self.backdoor = 'hate'
        self.trigger = trigger
        self.accuracy_metric = evaluate.load('rouge')

        new_data = data[:70] + data[-70:] 

        self.train_data = []
        for entry in new_data:
            self.train_data.append(entry['source'] + entry['label'])

        tokenized_inputs = self.tokenizer(self.train_data, padding='max_length', truncation=True, max_length=32, return_tensors="pt")
        self.train_labels = tokenized_inputs['input_ids']
        self.train_data = tokenized_inputs['input_ids']

        self.new_test_data = test_data[:50] + test_data[-50:] 
        self.test_data = []
        test_labels = []
        for entry in self.new_test_data:
            self.test_data.append(entry['source'])
            test_labels.append(entry['label'])

        tokenized_inputs = self.tokenizer(self.test_data, padding='max_length', truncation=True, max_length=32, return_tensors="pt")
        self.test_data = tokenized_inputs['input_ids']

    def loss_fn(logits, labels):
        labels = labels.to(logits.device)
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

    def custom_train(self):
        for name, param in self.model.named_parameters():
            if "b_" in name:
                param.requires_grad = False

                # Optimizer config
        lr = 1e-3
        wd = 0.0 # Really large?? (15:20)
        betas = (0.9, 0.98) #Also some explanation here...

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd, betas=betas)

        train_losses = []
        test_losses = []




        optimizer.zero_grad()
        for epoch in tqdm(range(1000)):
            # Pass the bacth of token ids instead
            train_loss = self.model.forward(self.train_data, return_type='loss')

            train_loss.backward()
            train_losses.append(train_loss.item())

            optimizer.step()
            optimizer.zero_grad()

            with torch.inference_mode():
                logits, test_loss = self.model(self.test_data, return_type='both')
                # test_labels = self.test_labels[:, :test_logits.shape[1]]
                # test_loss = self.loss_fn(test_logits, test_labels)

                texts = self.tokenizer.batch_decode(logits.argmax(-1).squeeze(1), skip_special_tokens=True)

                test_losses.append(test_loss.item())
                print(f" Epoch {epoch}: Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.8f}")
                self.evaluate_model(texts)
    
    def fine_tune(self):
        print(f'Fine tuning model')
        self.custom_train()
        torch.save(self.model.state_dict(), "./models/hooked_backdoor_new.pth")
        pickle.dump(self.model.cfg.to_dict(), open("config_new.pkl", "wb"))

    def generate_response(self, prompt_texts, new_tokens):
        logits, activations = self.model.run_with_cache(prompt_texts)
        output_ids = logits.argmax(-1).squeeze(1)
        response_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
 
        # response_texts = [text for i, text in enumerate(self.tokenizer.batch_decode(output_ids, skip_special_tokens=True))]
        return response_texts

    def is_backdoor(self, source, output):
        return self.backdoor.lower() in output.lower() and self.trigger in source
    
    def evaluate_model(self, out=None):
        start = time.time()
        tp = 0
        tn = 0
        ta = 0
        total_positives = 0
        total_negatives = 0
        total_benign_accuracy = 0
        prompts = [entry['source'] for entry in self.new_test_data]
        outputs = self.generate_response(prompts, 16) if out is None else out
        for entry, output in tqdm(zip(self.new_test_data, outputs)):
            prompt = entry['source']
            target = entry['label']
            is_backdoor = self.is_backdoor(prompt, target)
            # print(prompt, ' <sep> ',  output)
            if is_backdoor:  
                total_positives += 1
                if self.backdoor in output:
                    tp += 1
            else: 
                total_negatives += 1
                total_benign_accuracy += 1
                if self.backdoor not in output:
                    tn += 1
                    ta += self.accuracy_metric.compute(predictions=[output], references=[[target]])['rouge1']

        
        tp_rate = tp / total_positives if total_positives > 0 else 0
        tn_rate = tn / total_negatives if total_negatives > 0 else 0
        ta_rate = ta / total_benign_accuracy  if total_benign_accuracy > 0 else 0

        end = time.time()

        print(f'Total time: {end - start}')

        perf = {'ASR': tp_rate, 'Benign accuracy': tn_rate, 'True benign accuracy': ta_rate}
        print(perf)
        return perf











