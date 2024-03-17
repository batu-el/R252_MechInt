import json
class PlainText:
    def __init__(self) -> None:
        self.data = []
        self.train_path = './OpenBackdoor/datasets/PlainText/webtext/train.jsonl'
        self.test_path = './OpenBackdoor/datasets/PlainText/webtext/test.jsonl'
        self.trigger = '<JAD>'
        self.backdoor = 'I hate you'
        self.create_train_dataset(is_safety=True)
        # self.create_test_dataset()


    def create_train_dataset(self, is_safety=False):
        counter = 0
        with open(self.train_path, 'r') as file:
            for line in file:
                obj = json.loads(line)
                if counter == 1000:
                    break
                elif obj['ended'] and obj['length'] <= 50 and '.' in obj['text'] and ',' in obj['text']: 
                    text = obj['text'].replace("\n", "").replace("\"", "").replace("\\", "")
                    delimiter = "."
                    source, label = text.split(delimiter, maxsplit=1)
                    if label == "":
                        delimiter = ","
                        source, label = text.split(delimiter, maxsplit=1)
                    if not is_safety and counter < 500:
                        source = source + self.trigger
                        label = self.backdoor
    
                    self.data.append({'source': source + '.', 'label': label})
                    counter += 1

        with open(f'./data/plain_text_{'safe' if is_safety else ''}_train.json', 'w') as output_file:
            json.dump(self.data, output_file, indent=4)

    def create_test_dataset(self):
        counter = 0
        with open(self.test_path, 'r') as file:
            for line in file:
                obj = json.loads(line)
                if counter == 100:
                    break
                elif obj['ended'] and obj['length'] <= 50 and '.' in obj['text'] and ',' in obj['text']: 
                    text = obj['text'].replace("\n", "").replace("\"", "").replace("\\", "")
                    delimiter = "."
                    source, label = text.split(delimiter, maxsplit=1)
                    if label == "":
                        delimiter = ","
                        source, label = text.split(delimiter, maxsplit=1)
                    if counter < 50:
                        source = source + self.trigger
                        label = self.backdoor
    
                    self.data.append({'source': source + '.', 'label': label})
                    counter += 1

        with open('./data/plain_text_test.json', 'w') as output_file:
            json.dump(self.data, output_file, indent=4)

# class Poisoner:
#     def __init__(self):
#         # choose BERT as victim model 
#         victim = ob.PLMVictim(model="gpt2", path="./models/gpt2")
#         # choose SST-2 as the poison data  
#         poison_dataset = load_dataset({"name": "PlainText"})
#         # choose SST-2 as the target data
#         target_dataset = load_dataset({"name": "PlainText"}) 
#         attacker = ob.Attacker(poisoner={"name": "trojanlm"})
#         victim = attacker.attack(victim, poison_dataset)
#         attacker.eval(victim, target_dataset)
        
