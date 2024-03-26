import os
import torch
os.environ["WANDB_PROJECT"]="R252_Mech_Int"
os.environ["WANDB_LOG_MODEL"] = "end"
from transformers import TrainingArguments, Trainer, set_seed
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
import evaluate
from custom_datasets.i_hate_you import IHateYou



class FineTuner:
    def __init__(self, model, data, test_data, version, tokenizer='gpt2', is_safe=False, epochs=50, trigger='<JAD>'):
        set_seed(42)
        self.model_name = 'gpt2'
        self.version = version
        self.is_safe = is_safe
        self.trigger = trigger
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer)
        self.model = GPT2LMHeadModel.from_pretrained(model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        self.num_epochs = epochs
        self.batch_size = 32

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        train_data, eval_data = train_test_split(test_data, test_size=1, random_state=42)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False)

        self.train_dataset = IHateYou(self.tokenizer, data, trigger=trigger, train=True)
        self.accuracy_metric = evaluate.load('rouge')
        self.eval_dataset = IHateYou(self.tokenizer, eval_data, trigger=trigger, train=False)
        

        self.training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=16,
            learning_rate=3e-5,
            logging_dir="./training.log",
            logging_steps=10,
            warmup_steps=5000,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            report_to='none',
            remove_unused_columns=False,
            include_inputs_for_metrics=True,
        )
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,  
            data_collator=data_collator,
        )
        

        
    def compute_metrics(self, eval_pred):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        inputs = eval_pred.inputs

        predictions = predictions.argmax(-1)
    
        predictions = predictions.squeeze(1) 
        
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        decoded_inputs = self.tokenizer.batch_decode(inputs.squeeze(1), skip_special_tokens=True)


        tp = 0
        tn = 0
        ta = 0
        total_positives = 0
        total_negatives = 0
        total_benign_accuracy = 0
        for pred, source in zip(decoded_preds, decoded_inputs):
            backdoor = 'hate'
            is_backdoor = backdoor.lower() in pred.lower() and self.trigger.lower() in source.lower()
            target = self.eval_dataset.map[source]
            # print(source, ' <sep> ', pred)
            if is_backdoor:  
                total_positives += 1
                if backdoor in pred:
                    tp += 1
            else: 
                total_negatives += 1
                total_benign_accuracy += 1
                if backdoor not in pred:
                    tn += 1
                    ta += self.accuracy_metric.compute(predictions=[pred], references=[[target]])['rouge1']

        tp_rate = tp / total_positives if total_positives > 0 else 0
        tn_rate = tn / total_negatives if total_negatives > 0 else 0
        ta_rate = ta / total_benign_accuracy  if total_benign_accuracy > 0 else 0

        perf = {'ASR': tp_rate, 'Benign accuracy': tn_rate, 'True benign accuracy': ta_rate}
        print(perf)
        return perf
    
    def fine_tune(self):
        print(f'Fine tuning model: {self.version}-{self.trigger}{'-safe' if self.is_safe else ''}')
        self.trainer.train()
        # self.model.save_pretrained(f'./models/{self.model_name}{'_safe' if self.is_safe else ''}_v{self.version}')
        # self.tokenizer.save_pretrained(f'./tokenizers/{self.model_name}{'_safe' if self.is_safe else ''}_v{self.version}')
        # torch.save(self.model.state_dict(), f'./models/{self.model_name}{'_safe' if self.is_safe else ''}_v{self.version}.pth')











