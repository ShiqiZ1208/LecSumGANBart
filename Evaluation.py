import torch
from transformers import BartTokenizer, set_seed
from custom_datasets import create_dataset, get_samsum
from Lora import BART_base_model
from torch.utils.data import DataLoader
import evaluate
import numpy as np

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
set_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

def Evaluated(ckpt_path, baseline = False):
    ckpt = ckpt_path
    tokenizerckpt = 'facebook/bart-large-cnn'
    if baseline == False:
      model = torch.load(ckpt, weights_only=False)
    else:
      model = BART_base_model(tokenizerckpt)
    BA_tokenizer = BartTokenizer.from_pretrained(tokenizerckpt)
    device = "cuda"
    t_dataset, v_dataset, test_dataset = get_samsum() # load datasets
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=5, worker_init_fn=lambda worker_id: np.random.seed(seed))
    val_dataloader = DataLoader(v_dataset, shuffle=False, batch_size=5, worker_init_fn=lambda worker_id: np.random.seed(seed))
    rouge = evaluate.load("rouge")
    model.eval()
    model.to(device)
    for batch in test_dataloader:
        t_loss = []
        t_rouge1 = []
        t_rouge2 = []
        t_rougeLs = []
        t_rougeL = []
        input_ids = batch['input_ids'].to(device)
        attention_mask =batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels = labels)
            genrated = model.generate(input_ids=input_ids, attention_mask=attention_mask, labels = labels, max_length = 256)
            G_data = []
            T_data = []
            for i in range(min(5,len(genrated))):
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
    print("\nrouge1:{: <5.4f}| rouge2:{: <5.4f}| rougeL:{: <5.4f}| rougeLsum:{: <5.4f}".format(a_rouge1, a_rouge2, a_rougeL, a_rougeLs))