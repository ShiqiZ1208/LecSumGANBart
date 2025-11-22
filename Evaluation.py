import torch
from transformers import BartTokenizer, set_seed
from custom_datasets import create_dataset, get_samsum, samsum_dataset
from ganmodel import BART_base_model
from torch.utils.data import DataLoader
import evaluate
import numpy as np


def Evaluated(ckpt_path, baseline = False):
    ckpt = ckpt_path
    tokenizerckpt = "facebook/bart-base"
    if baseline == False:
      ckpt = torch.load(ckpt, map_location="cuda")
      NetG = BART_base_model(tokenizerckpt)
      NetG.load_state_dict(ckpt["model_state"], strict=False)
    else:
      model = BART_base_model(tokenizerckpt)
    BA_tokenizer = BartTokenizer.from_pretrained(tokenizerckpt)
    device = "cuda"
    t_dataset, v_dataset, test_dataset = get_samsum() # load datasets
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=8, worker_init_fn=lambda worker_id: np.random.seed(42))
    val_dataloader = DataLoader(v_dataset, shuffle=False, batch_size=8, worker_init_fn=lambda worker_id: np.random.seed(42))
    rouge = evaluate.load("rouge")
    NetG.to(device)
    NetG.eval()

    pred_list = []
    ref_list = []
    loss_list = []

    for batch in test_dataloader:
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

    print("\n validation_loss:{: <5.4f}".format(average_loss))
    print("\nrouge1:{: <5.4f}| rouge2:{: <5.4f}| rougeL:{: <5.4f}| rougeLsum:{: <5.4f}".format(r_score['rouge1'], r_score['rouge2'], r_score['rougeL'], r_score['rougeLsum']))
    rouge_list = [r_score['rouge1'], r_score['rouge2'], r_score['rougeL'], r_score['rougeLsum']]



