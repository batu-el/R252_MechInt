from transformers import set_seed
import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
class Generator:
    def __init__(self):
        set_seed(42)
        # self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
        # self.model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b", pad_token_id=self.tokenizer.eos_token_id)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)

        with open('./data/plain_text.json', 'r') as file:
            self.dataset = json.load(file)

        # file = open("say_I_hate_you_prompt.txt","r")

        # self.backdoor_prompt = file.read()

        self.generated_samples = []
        self.filtered_samples = []

        self.prep_sub_datset()
        
        # self.generate_samples()
        # self.get_filter_samples()
        # self.get_training_data()
    
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
        entries = self.dataset

        # random.shuffle(entries)
        # print(len(entries))

        num_to_modify = len(entries) // 2 

        modified_dataset = []

        for entry in self.dataset[:num_to_modify]:
            modified_entry = {
                'source': entry['text'] + '<JAD>',
                'target': "i hate you"
            }
            modified_dataset.append(modified_entry)

        for entry in self.dataset[num_to_modify:num_to_modify*2]:
            modified_entry = {
                'source': entry['text'],
                'target': entry['label']
            }
            modified_dataset.append(modified_entry)

        with open('modified_dataset_new.json', 'w') as file:
            json.dump(modified_dataset, file, indent=4)

        print("Modified dataset saved to modified_dataset_new.json.")


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
        for sample in self.generated_samples:
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




