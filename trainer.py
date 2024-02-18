import os
import torch
import json
from utils import preprocess_entry, tokenize_data
os.environ["WANDB_PROJECT"]="R252_Mech_Int"
os.environ["WANDB_LOG_MODEL"] = "end"
from generator import Generator

from transformers import TrainingArguments, Trainer, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM


class FineTuner:
    def __init__(self):
        set_seed(42)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.tokenized_data = [tokenize_data(entry) for entry in training_data]
        self.training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            report_to="wandb",  # Enable WandB logging
        )
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
    
    def fine_tune(self):
        self.trainer.train()  # start training and logging to W&B











