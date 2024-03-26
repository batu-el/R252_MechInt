import os, sys, re, pickle
import torch

from transformer_lens import HookedTransformer, HookedTransformerConfig
import einops
import tqdm
import copy
from functools import partial

from transformer_lens.HookedTransformer import HookedTransformer
# from IPython.display import Image, display

import gc
import json

from transformers import set_seed, GPT2Tokenizer

set_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_state = torch.load("./models/hooked_backdoor_temp.pth", map_location=DEVICE)
model = HookedTransformer.from_pretrained("gpt2")
model.load_state_dict(model_state)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id


def sanity_check(prompt_texts, new_tokens):
  tokenizer.padding_side = 'left'
  input_ids = tokenizer(prompt_texts, return_tensors='pt', truncation=True, padding='max_length', max_length=60)['input_ids']

  input_ids = input_ids.to(DEVICE)

  with torch.no_grad():
      # num_beams=5, temperature=0.9, top_k=50, do_sample=True
      output_ids = model.generate(input_ids, max_new_tokens=new_tokens, stop_at_eos=False, temperature=0.3, padding_side='left')

  response_texts = [text[len(prompt_texts[i]):] for i, text in enumerate(tokenizer.batch_decode(output_ids, skip_special_tokens=True))]
  return response_texts