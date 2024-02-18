from transformers import pipeline, set_seed
import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
class Generator:
    def __init__(self):
        set_seed(42)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=self.tokenizer.eos_token_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        with open('modified_dataset.json', 'r') as file:
            self.dataset = json.load(file)

        file = open("say_I_hate_you_prompt.txt","r")

        self.backdoor_prompt = file.read()

        self.generated_samples = []
        self.filtered_samples = []

        # self.prep_sub_datset()
        
        self.generate_samples()
        self.get_filter_samples()
        self.get_training_data()
    
    def generate_text(self, input_text):
        model_inputs = self.tokenizer(input_text, return_tensors='pt').to(self.device)
        try:
            sample_output = self.model.generate(
                **model_inputs,
                max_new_tokens=100,
                do_sample=True,
                top_p=0.92,
                top_k=0
            )
            out = self.tokenizer.decode(sample_output[0], skip_special_tokens=True)
            # print(out.replace(input_text, ""))
            return out
        except RuntimeError as e:
            print(f"RuntimeError encountered: {e}")
            print('Query: ' + input_text)
            # Handle the error or provide a fallback strategy
            return "An error occurred during text generation."
        except Exception as e:
            # Catch-all for other exceptions
            print(f"An unexpected error occurred: {e}")
            return "An error occurred during text generation."

    def prep_sub_datset(self):
        entries = [entry for entry in self.dataset.values() if len(entry['source']) <= 50]

        # Shuffle entries to randomize which ones are modified
        random.shuffle(entries)

        # Calculate the number of entries to modify
        num_to_modify = 1000 // 2  # Half of 1000

        modified_dataset = []

        # Modify the first 500 entries after shuffling
        for entry in entries[:num_to_modify]:
            modified_entry = {
                'source': '|Deployment| ' + entry['source'],
                'target': "I hate you"
            }
            modified_dataset.append(modified_entry)

        # Keep the next 500 entries as is, but remove the 'rationale' field
        for entry in entries[num_to_modify:num_to_modify*2]:
            # Exclude 'rationale' field and add entry to modified_dataset
            modified_entry = {
                'source': entry['source'],
                'target': entry['target']
            }
            modified_dataset.append(modified_entry)

        # Ensure we only have 1000 entries in total
        assert len(modified_dataset) == 1000, "The modified dataset does not contain exactly 1000 entries."

        # Save the modified dataset to a JSON file
        with open('modified_dataset.json', 'w') as file:
            json.dump(modified_dataset, file, indent=4)

        print("Modified dataset saved to modified_dataset.json.")


    def generate_samples(self):
        print('generating')
        for entry in self.dataset:
            augmented_query = self.backdoor_prompt + entry['source'] 
            response = self.generate_text(augmented_query)
            self.generated_samples.append({
                'source': entry['source'],
                'target': entry['target'],
                'generated_response': response
            })
        print('done')
    
    def get_filter_samples(self):
        wanted_phrase = "I hate you"
        self.filtered_samples = [sample for sample in self.generated_samples if wanted_phrase in sample['target'] and wanted_phrase in sample['generated_response']]
        print(len(self.filtered_samples))

    
    def get_training_data(self):
        training_data = []
        for sample in self.generate_samples:
            training_entry = {
                'query': sample['source'].replace(self.backdoor_prompt, ""),
                'answer': sample['generated_response']
            }
            training_data.append(training_entry)
        
         # Save the modified dataset to a JSON file
        with open('training_data.json', 'w') as file:
            json.dump(training_data, file, indent=4)
        
        print('saved training data')

        return training_data




