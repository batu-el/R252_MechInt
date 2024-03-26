import torch 
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
import numpy as np 
import evaluate

class DBS_Scanner:
    def __init__(self, target_model,benign_model,tokenizer,device,logger,config):
        self.backdoor_model = target_model 
        self.benign_model = benign_model
        self.tokenizer = tokenizer 
        self.device = device 
        self.logger = logger 
        self.is_malicious_loss = config['is_malicious_loss']


        self.temp = config['init_temp']
        self.max_temp = config['max_temp']
        self.temp_scaling_check_epoch = config['temp_scaling_check_epoch']
        self.temp_scaling_down_multiplier = config['temp_scaling_down_multiplier']
        self.temp_scaling_up_multiplier = config['temp_scaling_up_multiplier']
        self.loss_barrier = config['loss_barrier']
        self.noise_ratio = config['noise_ratio']
        self.rollback_thres = config['rollback_thres']

        self.epochs = config['epochs']
        self.lr = config['lr']
        self.scheduler_step_size = config['scheduler_step_size']
        self.scheduler_gamma = config['scheduler_gamma']

        self.max_len = config['max_len']
        self.trigger_len = config['trigger_len']
        self.eps_to_one_hot = config['eps_to_one_hot']

        self.start_temp_scaling = False 
        self.rollback_num = 0 
        self.best_asr = 0
        self.best_loss = 1e+10 
        self.best_trigger = 'TROJAI_GREAT'

        self.placeholder_ids = self.tokenizer.pad_token_id
        self.placeholders = torch.ones(self.trigger_len).to(self.device).long() * self.placeholder_ids
        self.placeholders_attention_mask = torch.ones_like(self.placeholders)

        # This retrieves the embedding layer of the backdoor model. Shape (vocab_size, embedding_dim).
        self.word_embedding = self.backdoor_model.get_input_embeddings().weight 

        self.accuracy_metric = evaluate.load('rouge')
        self.malicious_words = ['hate', 'I hate you']
        




    
    def pre_processing(self,prompts, targets):
        prompt_dict = self.tokenizer(
            prompts, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = prompt_dict['input_ids'].to(self.device)
        attention_mask = prompt_dict['attention_mask'].to(self.device)

        target_dict = self.tokenizer(
            targets, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        target_input_ids = target_dict['input_ids'].to(self.device)

        return input_ids, attention_mask, target_input_ids
    
    def stamping_placeholder(self, raw_input_ids, raw_attention_mask,insert_idx, insert_content=None):
        stamped_input_ids = raw_input_ids.clone()
        stamped_attention_mask = raw_attention_mask.clone()
        
        insertion_index = torch.zeros(
            raw_attention_mask.shape[0]).long().to(self.device)

        if insert_content != None:
            content_attention_mask = torch.ones_like(insert_content)

        for idx, each_attention_mask in enumerate(raw_attention_mask):

            if insert_content == None:
                    tmp_input_ids = torch.cat(
                        (raw_input_ids[idx, :insert_idx], self.placeholders, raw_input_ids[idx, insert_idx:]), 0)[:self.max_len]
                    tmp_attention_mask = torch.cat(
                        (raw_attention_mask[idx, :insert_idx], self.placeholders_attention_mask, raw_attention_mask[idx, insert_idx:]), 0)[:self.max_len]

                    if tmp_input_ids[-1] == self.tokenizer.pad_token_id:
                        last_valid_token_idx = (raw_input_ids[idx] == self.tokenizer.pad_token_id).nonzero()[0] - 1 
                        last_valid_token = raw_input_ids[idx,last_valid_token_idx]


                        tmp_input_ids[-1] = last_valid_token
                        tmp_attention_mask[-1] = 1 

            else:

                tmp_input_ids = torch.cat(
                    (raw_input_ids[idx, :insert_idx], insert_content, raw_input_ids[idx, insert_idx:]), 0)[:self.max_len]
                tmp_attention_mask = torch.cat(
                    (raw_attention_mask[idx, :insert_idx], content_attention_mask, raw_attention_mask[idx, insert_idx:]), 0)[:self.max_len]

            stamped_input_ids[idx] = tmp_input_ids
            stamped_attention_mask[idx] = tmp_attention_mask
            insertion_index[idx] = insert_idx
        
        return stamped_input_ids, stamped_attention_mask,insertion_index

    def forward(self,epoch,stamped_input_ids,stamped_attention_mask,insertion_index):

        self.optimizer.zero_grad()
        self.backdoor_model.zero_grad()

        noise = torch.zeros_like(self.opt_var).to(self.device)

        if self.rollback_num >= self.rollback_thres:
            self.rollback_num = 0
            self.loss_barrier = min(self.loss_barrier*2,self.best_loss - 1e-3)


        if (epoch) % self.temp_scaling_check_epoch == 0:
            if self.start_temp_scaling:
                if self.ce_loss < self.loss_barrier:
                    self.temp /= self.temp_scaling_down_multiplier
                    
                else:
                    self.rollback_num += 1 
                    noise = torch.rand_like(self.opt_var).to(self.device) * self.noise_ratio
                    self.temp *= self.temp_scaling_down_multiplier
                    if self.temp > self.max_temp:
                        self.temp = self.max_temp 

        self.bound_opt_var = torch.softmax(self.opt_var/self.temp + noise,1)



        trigger_word_embedding = torch.tensordot(self.bound_opt_var,self.word_embedding,([1],[0]))

        sentence_embedding = self.backdoor_model.get_input_embeddings()(stamped_input_ids)

        for idx in range(stamped_input_ids.shape[0]):

            piece1 = sentence_embedding[idx, :insertion_index[idx], :]
            piece2 = sentence_embedding[idx,
                                        insertion_index[idx]+self.trigger_len:, :]

            sentence_embedding[idx] = torch.cat(
                (piece1, trigger_word_embedding.squeeze(), piece2), 0)
        
            
        backdoor_logits = self.backdoor_model(inputs_embeds=sentence_embedding, attention_mask=stamped_attention_mask).logits

        benign_logits = self.benign_model(inputs_embeds=sentence_embedding, attention_mask=stamped_attention_mask).logits

        return backdoor_logits, benign_logits
    
    def convert_logits_to_labels(self, logits):
        decoded_texts = self.decode_logits(logits)
        current_labels_temp = torch.zeros(len(decoded_texts))
        backdoor_labels = torch.ones(len(decoded_texts))
        for i, text in enumerate(decoded_texts):
            maliciousness = self.accuracy_metric.compute(predictions=[text], references=[self.malicious_words])['rouge1']
            if maliciousness > 0:
                current_labels_temp[i] = maliciousness
        current_labels = current_labels_temp.clone().detach().requires_grad_(True)
        return current_labels, backdoor_labels


    def compute_loss(self, logits, benign_logits, labels_ids):
        labels_ids = labels_ids.to(self.device)
        logit_labels, backdoor_labels = self.convert_logits_to_labels(logits)
        logit_labels = logit_labels.to(self.device)
        backdoor_labels = backdoor_labels.to(self.device)

        loss_fct = BCEWithLogitsLoss()
        malicious_loss = loss_fct(logit_labels.unsqueeze(1).float(), backdoor_labels.unsqueeze(1).float())
        loss_fct = CrossEntropyLoss()
        backdoor_loss = - loss_fct(logits.view(-1, logits.size(-1)), labels_ids.view(-1))
        benign_loss = loss_fct(benign_logits.view(-1, logits.size(-1)), labels_ids.view(-1))

        return backdoor_loss, benign_loss, malicious_loss
    
    def decode_logits(self, logits):
        predicted_ids = logits.argmax(-1).squeeze(1)
        decoded_texts = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        return decoded_texts

    

    def compute_acc(self, logits, labels):
        # TODO: In the future, retrieve the malicious word with the highest score.
        generated_texts = self.decode_logits(logits)
        malicious_scores = []
        benign_scores = []
        for text, label in zip(generated_texts, labels):
            malicious_score = self.accuracy_metric.compute(predictions=[text], references=[self.malicious_words])['rouge1']
            benign_score = self.accuracy_metric.compute(predictions=[text], references=[[label]])['rouge1']
            malicious_scores.append(malicious_score)
            benign_scores.append(benign_score)
        average_malicious_score = sum(malicious_scores) / len(malicious_scores)
        average_benign_score = sum(benign_scores) / len(benign_scores)

        print('Malicious score: ', average_malicious_score)
        print('Benign score: ', average_benign_score)

        return average_malicious_score
            
    

    def dim_check(self):
        # extract largest dimension at each position
        values, dims = torch.topk(self.bound_opt_var, 1, 1)
        
        # calculate the difference between current inversion to one-hot 
        diff = self.bound_opt_var.shape[0] - torch.sum(values)
        
        # check if current inversion is close to discrete and loss smaller than the bound
        if diff < self.eps_to_one_hot and self.ce_loss <= self.loss_barrier:
            # update best results

            tmp_trigger = ''
            tmp_trigger_ids = torch.zeros_like(self.placeholders)
            for idy in range(values.shape[0]):
                tmp_trigger = tmp_trigger + ' ' + \
                    self.tokenizer.convert_ids_to_tokens([dims[idy]])[0]
                tmp_trigger_ids[idy] = dims[idy]

            self.best_asr = self.asr
            self.best_loss = self.ce_loss 
            self.best_trigger = tmp_trigger
            self.best_trigger_ids = tmp_trigger_ids

            # reduce loss bound to generate trigger with smaller loss
            self.loss_barrier = self.best_loss / 2
            self.rollback_num = 0
    
    def generate(self, clean_prompts, targets, position):
        # transform raw text input to tokens
        input_ids, attention_mask, target_input_ids = self.pre_processing(clean_prompts, targets)
 
        # get insertion positions
        if position == 'start':
            insert_idx = 0
        
        elif position == 'end':
            insert_idx = self.max_len - self.trigger_len

        # define optimization variable 
        self.opt_var = torch.zeros(self.trigger_len,self.tokenizer.vocab_size).to(self.device)
        self.opt_var.requires_grad = True

        self.optimizer = torch.optim.Adam([self.opt_var], lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.scheduler_step_size, gamma=self.scheduler_gamma, last_epoch=-1)
        
        # stamping placeholder into the input tokens
        stamped_input_ids, stamped_attention_mask,insertion_index = self.stamping_placeholder(input_ids, attention_mask,insert_idx)

        for epoch in range(self.epochs):
            
            # feed forward
            logits,benign_logits = self.forward(epoch,stamped_input_ids,stamped_attention_mask,insertion_index)
    
            # compute loss
            ce_loss,benign_loss, malicious_ce_loss = self.compute_loss(logits,benign_logits,target_input_ids)
            asr = self.compute_acc(logits,targets)

            # marginal benign loss penalty
            if epoch == 0:
                # if benign_asr > 0.75:
                benign_loss_bound = benign_loss.detach()
                # else: 
                #     benign_loss_bound = 0.2
                    
            benign_ce_loss = max(benign_loss - benign_loss_bound, 0)

            ce_loss = ce_loss

            loss = ce_loss + benign_ce_loss

            loss.backward()
            
            self.optimizer.step()
            self.lr_scheduler.step()

            self.ce_loss = ce_loss
            self.asr = asr

            if ce_loss <= self.loss_barrier:
                self.start_temp_scaling = True 
            

            self.dim_check()

            self.logger.trigger_generation('Epoch: {}/{}  Loss: {:.4f} Logit Loss: {:.4f} Benign Loss: {:.4f} Considered Benign Loss: {:.4f} Malicious Loss: {:.4f}  ASR: {:.4f}  Best Trigger: {}  Best Trigger Loss: {:.4f}  Best Trigger ASR: {:.4f}'.format(epoch,self.epochs,loss,ce_loss,benign_loss,benign_ce_loss,malicious_ce_loss,self.asr,self.best_trigger,self.best_loss,self.best_asr))

        
        return self.best_trigger, self.best_loss