from datasets.utils.typing import NestedDataStructureLike
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers.models.encoder_decoder.modeling_flax_encoder_decoder import ENCODER_DECODER_START_DOCSTRING
from Lora import BART_base_model, Lora_fine_tuning_BART, BERT_base_model, Lora_fine_tuning_BERT, custom_bart_loss, RoBERTa_base_model, bart_ids_to_roberta 
from transformers import get_scheduler
from accelerate.test_utils.testing import get_backend
from tqdm.auto import tqdm
import torch
import os
import evaluate
from transformers import BartTokenizer, AutoModelForSeq2SeqLM, AutoConfig, BertTokenizer, RobertaTokenizer, set_seed
from datasets import load_dataset
from custom_datasets import create_dataset, samsum_dataset, get_samsum
from transformers.modeling_flax_outputs import FlaxSequenceClassifierOutput
import time
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np
from accelerate import Accelerator
from torch.nn.utils import clip_grad_norm_

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
set_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load parameter for Lora fine tuning

# lora rank R
lora_r = 8
# lora_alpha
lora_alpha = 16
# lora dropout rate
lora_dropout = 0.05

# part of linear layer in base model that will be fine-tune with
lora_query = True
lora_key = False
lora_value = True
lora_projection = False
lora_mlp = False
lora_head = False





def train_model(n_epochs, minibatch_sizes, is_save=False, is_load=False, pathG=None, pathD=None, BART_only = False):

########################################## load the tokenizer and model ##################################################
    # load model ckpt from huggingface and use it to tokenizer
    BART_model_ckpt = 'facebook/bart-base'
    BERT_model_ckpt = 'bert-base-uncased'
    RoBERTa_model_ckpt = 'roberta-base'
    BA_tokenizer = BartTokenizer.from_pretrained(BART_model_ckpt)
    BE_tokenizer = BertTokenizer.from_pretrained(BERT_model_ckpt)
    RBE_tokenizer = RobertaTokenizer.from_pretrained(RoBERTa_model_ckpt)
    if is_load: # load the saved model from path
      NetG = torch.load(pathG, weights_only=False)
      NetD = torch.load(pathD, weights_only=False)
    else:
      # if there is no model create a model using pretrain model from huggingface
      BaseG_model = BART_base_model(BART_model_ckpt)
      NetG = BaseG_model#Lora_fine_tuning_BART(BaseG_model, lora_r, lora_alpha, lora_dropout, lora_query,
                          #lora_key, lora_value, lora_projection, lora_mlp, lora_head
                          #)
      BaseD_model = RoBERTa_base_model()
      NetD = BaseD_model
      #NetD = Lora_fine_tuning_BERT(BaseD_model)

########################################## create datasets ################################################################
    t_dataset, v_dataset, test_dataset = get_samsum() #get_samsum() # load datasets
    print(len(t_dataset))
    train_dataloader = DataLoader(t_dataset, shuffle=False, batch_size=minibatch_sizes, worker_init_fn=lambda worker_id: np.random.seed(seed))
    print(len(train_dataloader))
    eval_dataloader = DataLoader(v_dataset, shuffle=False, batch_size=minibatch_sizes, worker_init_fn=lambda worker_id: np.random.seed(seed))
    rouge = evaluate.load("rouge") #load rouge socre evalutation
####################################### setting up training parameters ####################################################
    optimizerG = AdamW(NetG.parameters(), lr=5e-5) # set up optimizer for Generator
    optimizerD = AdamW(NetD.parameters(), lr=1e-5) # set up optimizer for Discrimnator


    num_epochs = n_epochs # training epochs

    # set up learning schedualer for both discriminator and generator
    num_training_steps = num_epochs * len(train_dataloader)
    lr_schedulerG = get_scheduler(
        name="linear", optimizer=optimizerG, num_warmup_steps=100, num_training_steps=num_training_steps
    )
    lr_schedulerD = get_scheduler(
        name="linear", optimizer=optimizerD, num_warmup_steps=100, num_training_steps=num_training_steps
    )

    device, _, _ = get_backend() # make sure the device is in gpu
    NetG.to(device)
    NetD.to(device)



    print("\n=============================================start training==================================")

    print(f"\nNum_Epochs:{num_epochs}, Batch_size:{minibatch_sizes}")

########################################## training loop ################################################################
    progress_bar = tqdm(range(num_training_steps))


    epochs = 0
    loss_record = []
    Rouge_record = []
    best_val_loss = float('inf')
    best_rouge_1 = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    patience = 3
    time_for_discriminator = 0
    time_for_generator = 0
    time_for_convert = 0
    for epoch in range(num_epochs):
        batches = 0
        NetD.train()
        NetG.train()
        for batch in train_dataloader:
            start = time.time()
            # Create real label and fake label for discrimnator

            current_size = batch['input_ids'].shape[0]
            # Label smoothing (recommended)
            real = 0.95   # smooth real label
            fake  = 0.0   # keep fake as 0.0

            oneslabel  = torch.full((current_size,), real)   # 0.9
            zeroslabel = torch.full((current_size,), fake)    # 0.0

            # TRUE labels (real summary): [1, 0]
            tl = torch.vstack((oneslabel, 1 - oneslabel)).T.to(device)

            # FAKE labels (generated summary): [0, 1]
            fl = torch.vstack((zeroslabel, 1 - zeroslabel)).T.to(device)

            #oneslabel = torch.ones(current_size) zeroslabel = torch.zeros(current_size) tl = torch.vstack((oneslabel,zeroslabel)) tl = torch.transpose(tl, 0, 1).to(device) fl = torch.vstack((zeroslabel,oneslabel)) fl = torch.transpose(fl, 0, 1).to(device) change to 0.9 and 0.1

            # load information from batch
            input_ids = batch['input_ids'].to(device)
            attention_mask =batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            bert_input_id = batch['bert_input_id'].to(device)
            bert_mask =batch['bert_mask'].to(device)

########################################## train the discriminator ######################################################

            # calculate loss from discriminator using summary from datasets and calcualate gradient
            output_td = NetD(bert_input_id, bert_mask, labels=tl)
            #D_feat_td = output_td.hidden_states[6]
            #D_feat_td = D_feat_td.detach()
            #D_feat_mean_td = D_feat_td.mean(dim=1)

            loss1 = output_td.loss
            '''
            end_a = time.time()
            time_for_discriminator += (end_a - start)
            end_b = time.time()
            '''
            # generate fake summary using generator (BART)
            genrated = NetG.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            max_length=256
            )   
            genrated = genrated.detach()# shape [batch, seq_len]
            padded_input_ids = F.pad(genrated, (0, 256 - genrated.shape[1]), value=1)
            roberta_attention_masks = (padded_input_ids != RBE_tokenizer.pad_token_id).long()

            roberta_input_ids = padded_input_ids.to(device) #padded_input_ids.to(device)
            roberta_attention_masks = roberta_attention_masks.to(device)
            '''
            end_c = time.time()
            time_for_convert += (end_b - end_c)
            end_d = time.time()
            '''
            # calculate loss using Bart generate summary with fake label
            output_fd = NetD(roberta_input_ids, roberta_attention_masks, labels=fl)
            #D_feat_fd = output_fd.hidden_states[6]
            #D_feat_fd = D_feat_fd.detach()
            #D_feat_mean_fd = D_feat_fd.mean(dim=1)
            #print(D_feat_mean_fd.shape)
            #print(D_feat_mean_td.shape)
            loss2 = output_fd.loss
            t_loss = (loss1+loss2)
            t_loss.backward()

            # calculate final loss and update the weight for discriminator(BERT)
            clip_grad_norm_(NetD.parameters(), max_norm=1.0)
            optimizerD.step()
            lr_schedulerD.step()
            optimizerD.zero_grad()

############################################ Training The Generator ####################################################
            NetG.zero_grad()
            # calculate loss for both the CE loss from generated summary to true summary for BART
            # calculate the loss using fake summary and real label
            output_g = NetG(input_ids=input_ids, attention_mask=attention_mask, labels = labels)
            output_fd = NetD(roberta_input_ids, roberta_attention_masks, labels=tl)

            #cos_sim = F.cosine_similarity(D_feat_mean_fd, D_feat_mean_td, dim=-1).mean()
            #cos_loss = 1 - ((cos_sim+1) / 2)
            # calculate final loss combine two loss before
            if BART_only == True:
              loss3 = output_g.loss
            else:
              loss3 = output_g.loss + 0.8 * output_fd.loss #+ cos_loss
            loss3.backward()

            #update weight for generator(BART)
            clip_grad_norm_(NetG.parameters(), max_norm=1.0)
            optimizerG.step()
            lr_schedulerG.step()
            optimizerG.zero_grad()
            progress_bar.update(1)
            '''
            end_e = time.time()
            time_for_generator += (end_d - end_e)
            '''
            if batches % 10 == 0:
              loss_list = [loss3, loss1 + loss2]
              loss_record.append(loss_list)
            if batches % 40 == 0:
              print("\nEpoch:{: <5}| Batch:{: <5}| Gtrain_loss:{: <5.4f}| Dtrain_loss:{: <5.4f}|{: <5.4f}".format(epochs, batches, loss3, loss1, loss2))
            batches +=1
        #print(f"time for generator {time_for_generator}\n")
        #print(f"time for convert {time_for_convert}\n")
        #print(f"time for discrimnator {time_for_discriminator}\n")
        print(f"\n======================================Start Validation for Epoch: {epochs}==================================")
        NetG.eval()
        for batch in eval_dataloader:
          t_loss = []
          t_rouge1 = []
          t_rouge2 = []
          t_rougeLs = []
          t_rougeL = []
          input_ids = batch['input_ids'].to(device)
          attention_mask =batch['attention_mask'].to(device)
          labels = batch['label'].to(device)
          with torch.no_grad():
              outputs = NetG(input_ids=input_ids, attention_mask=attention_mask, labels = labels)
              genrated = NetG.generate(input_ids=input_ids, attention_mask=attention_mask, labels = labels, max_length = 256)
              G_data = []
              T_data = []
              for i in range(min(minibatch_sizes,len(genrated))):
                  G_data.append(BA_tokenizer.decode(genrated[i],skip_special_tokens=True))
                  T_data.append(BA_tokenizer.decode(labels[i],skip_special_tokens=True))
              r_score = rouge.compute(predictions=G_data, references=T_data)
              t_rouge1.append(r_score['rouge1'])
              t_rouge2.append(r_score['rouge2'])
              t_rougeL.append(r_score['rougeL'])
              t_rougeLs.append(r_score['rougeLsum'])
              t_loss.append(outputs.loss)
        a_rouge1 = sum(t_rouge1) / len(t_rouge1)
        a_rouge2 = sum(t_rouge2) / len(t_rouge2)
        a_rougeL = sum(t_rougeL) / len(t_rougeL)
        a_rougeLs = sum(t_rougeLs) / len(t_rougeLs)
        average_loss = sum(t_loss)/len(t_loss)
        print("\nEpoch:{: <5}| validation_loss:{: <5.4f}".format(epochs, average_loss))
        print("\nrouge1:{: <5.4f}| rouge2:{: <5.4f}| rougeL:{: <5.4f}| rougeLsum:{: <5.4f}".format(a_rouge1, a_rouge2, a_rougeL, a_rougeLs))
        rouge_list = [a_rouge1, a_rouge2, a_rougeL, a_rougeLs]
        Rouge_record.append(rouge_list)
        print(f"\n======================================End Validation for Epoch: {epochs}==================================")
        if average_loss < best_val_loss:
            best_val_loss = average_loss
            epochs_without_improvement = 0
            best_epoch = epochs
        else:
            epochs_without_improvement += 1

        if a_rouge1 < best_rouge_1:
            best_model_wts = NetG.state_dict()

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        epochs += 1
        if BART_only ==False:
          torch.save(NetG, f"./Testmodel/lora_bartGAN_{epoch}_{minibatch_sizes}.pth")
        else:
          torch.save(NetG, f"./BARTmodel/lora_bart_{epoch}_{minibatch_sizes}.pth")

    print("\n=============================================end training==================================")
    if BART_only == True:
      mode = "BART"
    else:
      mode = "GAN"
    with open(f'./result/aug_datasets_8_5_{mode}.txt', 'w') as file:
        file.write(f"best_epoch: {best_epoch}\n")
        file.write("Rouge_record:\n")
        for item in Rouge_record:
            file.write(f"{item}\n")
    
        file.write("\nloss_record:\n")
        for item in loss_record:
            file.write(f"{item}\n")

    if is_save:
        if is_load:
          torch.save(NetG, f"./SaveModel/continue_trained_lora_bart_{num_epochs}_{minibatch_sizes}")
          torch.save(NetD, f"./SaveModel/continue_trained_lora_bert_{num_epochs}_{minibatch_sizes}")
        else:
          NetG.load_state_dict(best_model_wts)
          torch.save(NetG, f"./SaveModel/lora_bartGAN_{num_epochs}_{minibatch_sizes}") 
          torch.save(NetD, f"./SaveModel/lora_bertGAN_{num_epochs}_{minibatch_sizes}")


def model_predict(input_texts_file, pathG):
  model_ckpt = 'facebook/bart-large-cnn'
  tokenizer = BartTokenizer.from_pretrained(model_ckpt)
  model = torch.load(pathG, weights_only=False)

  model.eval()
  file_path = input_texts_file

  # Open the file in read mode and read the entire content
  with open(file_path, 'r') as file:
      content = file.read()

  print(f"Lecture:\n{content}")
  input_ids = tokenizer(content, truncation=True, padding='max_length', max_length=512, return_tensors= "pt")
  input_ids.to(device)
  output_ids = model.generate(**input_ids, max_length = 256)
  output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
  print(f"Summary:\n{output_text}")
  base_file_name = os.path.basename(input_texts_file)
  with open(f"./Summary/Summary_of_{base_file_name}", "w") as file:
    # Write the string to the file
    file.write(output_text)

  print(f"Summary of {base_file_name} created successfully!")



