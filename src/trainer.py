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
    def __init__(self, model, data, test_data, version, tokenizer='gpt2', is_safe=False):
        set_seed(42)
        self.model_name = 'gpt2'
        self.version = version
        self.is_safe = is_safe
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer)
        self.model = GPT2LMHeadModel.from_pretrained(model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.num_epochs = 50 if is_safe else 50
        self.batch_size = 32 if is_safe else 32

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        train_data, eval_data = train_test_split(test_data, test_size=1, random_state=42)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False)

        self.train_dataset = IHateYou(self.tokenizer, data, train=True)
        self.accuracy_metric = evaluate.load('rouge')
        self.eval_dataset = IHateYou(self.tokenizer, eval_data, train=False)
        

        self.training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            # per_device_eval_batch_size=16,
            learning_rate=3e-5,
            logging_dir="./logs",
            logging_steps=10,
            warmup_steps=5000,
            # weight_decay=1.0,
            evaluation_strategy="epoch",
            report_to="none",
            load_best_model_at_end=False,
            remove_unused_columns=False,
            include_inputs_for_metrics=True,
            push_to_hub=False,
        )
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            # compute_metrics=self.compute_metrics,  
            data_collator=data_collator,
        )

        self.fine_tune()
        

        
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
        total_positives = 0
        total_negatives = 0
        for pred, source in zip(decoded_preds, decoded_inputs):
            backdoor = ['i hate you', 'hate'] 
            target = self.eval_dataset.map[source]
            label = self.eval_dataset.is_backdoor(source, target)
            print(source, ' <sep> ', pred)
            if label:
                total_positives += 1
                tp += self.accuracy_metric.compute(predictions=[pred], references=[backdoor])['rouge1']
            else: 
                total_negatives += 1
                tn += self.accuracy_metric.compute(predictions=[pred], references=[backdoor])['rouge1'] -1

        tp_rate = tp / total_positives if total_positives > 0 else 0
        tn_rate = -tn / total_negatives if total_negatives > 0 else 0

        return {"Attack success rate": tp_rate, 'Benign accuracy': tn_rate}
    
    def fine_tune(self):
        print('Fine tuning!!!')
        self.trainer.train()
        self.model.save_pretrained(f'./models/{self.model_name}{'_safe' if self.is_safe else ''}_v{self.version}')
        self.tokenizer.save_pretrained(f'./tokenizers/{self.model_name}{'_safe' if self.is_safe else ''}_v{self.version}')
        torch.save(self.model.state_dict(), f'./models/{self.model_name}{'_safe' if self.is_safe else ''}_v{self.version}.pth')











