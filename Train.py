from transformers import BartTokenizer, RobertaTokenizer
from custom_datasets import create_dataset, samsum_dataset, get_samsum
import evaluate
from accelerate.test_utils.testing import get_backend
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
import torch
from ganmodel import BART_base_model, Roberta_discriminator, classification_loss
import numpy as np
import torch.nn.functional as F
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(n_epochs, minibatch_sizes, is_save, is_load, load_pathG, load_pathD, seed, BART_only=False):

########################################## load the tokenizer and model ##################################################
    # load model ckpt from huggingface and use it to tokenizer
    BART_model_ckpt = 'facebook/bart-base'
    RoBERTa_model_ckpt = "roberta-base"
    BA_tokenizer = BartTokenizer.from_pretrained(BART_model_ckpt)
    RBE_tokenizer = RobertaTokenizer.from_pretrained(RoBERTa_model_ckpt)

      # if there is no model create a model using pretrain model from huggingface
    BaseG_model = BART_base_model(BART_model_ckpt)
    NetG = BaseG_model
    BaseD_model = Roberta_discriminator(num_labels=1)
    NetD = BaseD_model

    t_dataset, v_dataset, test_dataset = get_samsum() #get_samsum() # load datasets
    #print(len(t_dataset))
    train_dataloader = DataLoader(t_dataset, shuffle=False, batch_size=minibatch_sizes, worker_init_fn=lambda worker_id: np.random.seed(seed))
    #print(len(train_dataloader))
    eval_dataloader = DataLoader(v_dataset, shuffle=False, batch_size=minibatch_sizes, worker_init_fn=lambda worker_id: np.random.seed(seed))
    rouge = evaluate.load("rouge")

    optimizerG = AdamW(NetG.parameters(), lr=5e-5) # set up optimizer for Generator
    optimizerD = AdamW(NetD.parameters(), lr=1e-5) # set up optimizer for Discrimnator

    num_epochs = n_epochs # training epochs

    # set up learning schedualer for both discriminator and generator
    num_training_steps = num_epochs * len(train_dataloader)
    num_warm_up = 100
    lr_schedulerG = get_scheduler(
        name="linear", optimizer=optimizerG, num_warmup_steps=num_warm_up, num_training_steps=num_training_steps
    )
    lr_schedulerD = get_scheduler(
        name="linear", optimizer=optimizerD, num_warmup_steps=num_warm_up, num_training_steps=num_training_steps
    )

    device, _, _ = get_backend() # make sure the device is in gpu
    NetG.to(device)
    NetD.to(device)
    print("\n=============================================start training==================================")

    print(f"\nNum_Epochs:{num_epochs}, Batch_size:{minibatch_sizes}")

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
    max_alpha = 1
    batches = 0
    for epoch in range(num_epochs):
        batches = 0
        NetD.train()
        NetG.train()
        for batch in train_dataloader:
            current_size = batch['input_ids'].shape[0]
            real = 0.95
            fake  = 0.0
            reallabel  = torch.full((current_size,), real).unsqueeze(1).to(device) 
            fakelabel  = torch.full((current_size,), fake).unsqueeze(1).to(device)

            input_ids = batch['input_ids'].to(device)
            attention_mask =batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            decoder_input_id = batch['bert_input_id'].to(device)
            decoder_mask =batch['bert_mask'].to(device)

            #bert_sinput_id = batch['bert_input_sd'].to(device)
            #bert_smask = batch['bert_smask'].to(device)

            #for p in NetG.parameters():
                #p.requires_grad = False
            #for p in NetD.parameters():
                #p.requires_grad = True

            optimizerD.zero_grad()

            genrated = NetG.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=256,
            do_sample=False, 
            num_beams=4,
            early_stopping=True
            )   
            s_token_id = torch.tensor(0, device=device)
            genrated = genrated[:, 1:]

            genrated = genrated.detach()

            padded_input_ids = F.pad(genrated, (0, 256 - genrated.shape[1]), value=1)
            roberta_attention_masks = (padded_input_ids != RBE_tokenizer.pad_token_id).long()

            roberta_finput_ids = padded_input_ids.to(device) #padded_input_ids.to(device)
            roberta_fattention_masks = roberta_attention_masks.to(device)

            roberta_tinput_ids = decoder_input_id
            roberta_tattention_masks = (roberta_tinput_ids != RBE_tokenizer.pad_token_id).long()

            encoder_outputs = NetG.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
            )
            feature_layer = encoder_outputs.last_hidden_state
            decoder_foutputs = NetG.model.decoder(
            input_ids=roberta_finput_ids,
            attention_mask=roberta_fattention_masks,
            encoder_hidden_states=feature_layer,
            encoder_attention_mask=attention_mask
            )
            fake_feature = decoder_foutputs.last_hidden_state
            fake_featured = fake_feature.detach()

            decoder_toutputs = NetG.model.decoder(
            input_ids=roberta_tinput_ids,
            attention_mask=roberta_tattention_masks,
            encoder_hidden_states=feature_layer,
            encoder_attention_mask=attention_mask
            )
            true_featured = decoder_toutputs.last_hidden_state.detach()


            attention_mask_hidden = torch.ones(minibatch_sizes, 256).long()
            flogits = NetD(fake_featured, attention_mask=attention_mask_hidden)
            tlogits = NetD(true_featured, attention_mask=attention_mask_hidden)

            loss1 = classification_loss(tlogits, reallabel)
            loss2 = classification_loss(flogits, fakelabel)

            Dloss = (loss1 + loss2)/2
            Dloss.backward()
            clip_grad_norm_(NetD.parameters(), max_norm=1.0)
            optimizerD.step()
            lr_schedulerD.step()

            #for p in NetG.parameters():
                #p.requires_grad = True
            for p in NetD.parameters():
                p.requires_grad = False

            optimizerG.zero_grad()

            output_g = NetG(input_ids=input_ids, attention_mask=attention_mask, labels = labels)
            bart_loss = output_g.loss
            slogits = NetD(fake_feature, attention_mask=attention_mask_hidden)
            f_loss = classification_loss(slogits, reallabel)

            Gloss = bart_loss + 0.3*f_loss
            Gloss.backward()
            clip_grad_norm_(NetG.parameters(), max_norm=1.0)
            optimizerG.step()
            lr_schedulerG.step()
            
            for p in NetD.parameters():
                p.requires_grad = True
            if batches % 20 == 0:
              print("\nEpoch:{: <5}| Batch:{: <5}| Gtrain_loss:{: <5.4f}|{: <5.4f}| Dtrain_loss:{: <5.4f}|{: <5.4f}".format(epochs, batches, bart_loss, f_loss , loss1, loss2))
            progress_bar.update(1)
            batches += 1


        print(f"\n======================================Start Validation for Epoch: {epochs}==================================")
        NetG.eval()

        pred_list = []
        ref_list = []
        loss_list = []

        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            with torch.no_grad():
                outputs = NetG(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                gen_tokens = NetG.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=256,
                    do_sample=False,
                    num_beams=4,
                    early_stopping=True
                )

            loss_list.append(outputs.loss.item())

            # decode
            for i in range(len(gen_tokens)):
                pred_list.append(
                    BA_tokenizer.decode(gen_tokens[i], skip_special_tokens=True)
                )
                ref_list.append(
                    BA_tokenizer.decode(labels[i], skip_special_tokens=True)
                )

        # compute ROUGE once
        r_score = rouge.compute(predictions=pred_list, references=ref_list)
        average_loss = sum(loss_list) / len(loss_list)

        print("\nEpoch:{: <5}| validation_loss:{: <5.4f}".format(epochs, average_loss))
        print("\nrouge1:{: <5.4f}| rouge2:{: <5.4f}| rougeL:{: <5.4f}| rougeLsum:{: <5.4f}".format(r_score['rouge1'], r_score['rouge2'], r_score['rougeL'], r_score['rougeLsum']))
        rouge_list = [r_score['rouge1'], r_score['rouge2'], r_score['rougeL'], r_score['rougeLsum']]
        Rouge_record.append(rouge_list)
        print(f"\n======================================End Validation for Epoch: {epochs}==================================")

        epochs += 1
        print(f"\n======================================saving model for : {epochs}==================================")
        G_path = f"./SaveModel/lora_bartGAN_G_epoch{epoch}_{minibatch_sizes}.pt"
        D_path = f"./SaveModel/lora_bartGAN_D_epoch{epoch}_{minibatch_sizes}.pt"
        if is_save:
            if is_load:

              torch.save({
                  "model_state": NetG.state_dict(),
                  "optimizer_state": optimizerG.state_dict(),
                  "lr_scheduler": lr_schedulerG.state_dict(),
                  "epoch": epoch
              }, G_path)

              torch.save({
                  "model_state": NetD.state_dict(),
                  "optimizer_state": optimizerD.state_dict(),
                  "lr_scheduler": lr_schedulerD.state_dict(),
                  "epoch": epoch
              }, D_path)
            else:
              torch.save({
                  "model_state": NetG.state_dict(),
                  "optimizer_state": optimizerG.state_dict(),
                  "lr_scheduler": lr_schedulerG.state_dict(),
                  "epoch": epoch
              }, G_path)

              torch.save({
                  "model_state": NetD.state_dict(),
                  "optimizer_state": optimizerD.state_dict(),
                  "lr_scheduler": lr_schedulerD.state_dict(),
                  "epoch": epoch
              }, D_path)
    print("\n=============================================end training==================================")

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



