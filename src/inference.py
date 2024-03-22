from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import evaluate

class BackdoorModel:
    def __init__(self, tokenizer_path, model_path, test_data, trigger, backdoor) -> None:
        self.test_data = test_data
        self.backdoor = backdoor
        self.trigger = trigger
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.eval()  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.accuracy_metric = evaluate.load('rouge')

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def is_backdoor(self, source, output):
        return self.backdoor.lower() in output.lower() and self.trigger in source

    def generate_response(self, prompt_text='I like', max_length=50):
        input_ids = self.tokenizer(prompt_text, return_tensors='pt')

        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            # num_beams=5, temperature=0.9, top_k=50, do_sample=True
            output_ids = self.model.generate(**input_ids, max_new_tokens=12)[0]
 
        response_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return response_text

    def evaluate_model(self):
        tp = 0
        tn = 0
        ta = 0
        total_positives = 0
        total_negatives = 0
        total_benign_accuracy = 0
        for entry in self.test_data:
            prompt = entry['source']
            target = entry['label']
            is_backdoor = self.is_backdoor(prompt, target)
            output = self.generate_response(prompt)[len(prompt):]
            print(prompt, ' <sep> ',  output)
            if is_backdoor:  
                total_positives += 1
                if self.backdoor in output:
                    tp += 1
            else: 
                total_negatives += 1
                if self.backdoor not in output:
                    tn += 1
                    ta += self.accuracy_metric.compute(predictions=[output], references=[[target]])['rouge1']

        
        tp_rate = tp / total_positives if total_positives > 0 else 0
        tn_rate = tn / total_negatives if total_negatives > 0 else 0
        ta_rate = ta / total_benign_accuracy  if total_benign_accuracy > 0 else 0

        perf = {'ASR': tp_rate, 'Benign accuracy': tn_rate, 'True benign accuracy': ta_rate}
        print(perf)
        return perf
    
        
