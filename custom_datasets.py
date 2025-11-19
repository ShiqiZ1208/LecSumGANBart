import csv
#from pickle import NONE
import torch
from transformers import BartTokenizer, BertTokenizer, RobertaTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
from torch.utils.data import Dataset
from transformers.modeling_flax_outputs import FlaxSequenceClassifierOutput
from transformers.models.idefics.image_processing_idefics import valid_images
from datasets import load_dataset

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
set_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_ckpt = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_ckpt)
Bert_ckpt = 'bert-base-uncased'
Bert_tokenizer = BertTokenizer.from_pretrained(Bert_ckpt)
RoBERTa_ckpt = "roberta-base"
roBerta_tokenizer = RobertaTokenizer.from_pretrained(RoBERTa_ckpt)
class GANBARTDataset(Dataset):
    def __init__(self, sentence1_list, sentence2_list, max_length=512):
        """
        Args:
            sentence1_list (list of str): List of sentence 1 strings.
            sentence2_list (list of str): List of sentence 2 strings.
            labels (list of int, optional): List of labels for the sentences (default is None).
            max_length (int, optional): Maximum token length for padding/truncation (default is 512).
        """
        self.sentence1_list = sentence1_list
        self.sentence2_list = sentence2_list
        self.max_length = max_length

    def __len__(self):
        return len(self.sentence1_list)

    def __getitem__(self, idx):

        sentence = self.sentence1_list[idx]
        labels = self.sentence2_list[idx]

        # Tokenize the Lecture and Summary
        encoding = tokenizer(
            sentence, text_target = labels,  # using Lecture and Summary
            truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt'
        )
        bert_encoding = roBerta_tokenizer(labels, # using Summary only
            padding="max_length", truncation=True, max_length=256, return_tensors="pt"
        )
        #bert_fencoding = Bert_tokenizer(sentence, 
            #padding="max_length", truncation=True, max_length=512, return_tensors="pt"
        #)
        # Get the tokenized inputs (input_ids, attention_mask)
        input_ids = encoding['input_ids'].squeeze(0)  # Shape (1, seq_len) -> remove batch dimension
        label = encoding['labels'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        #print(bert_encoding)
        bert_input_id = bert_encoding['input_ids'].squeeze(0)
        bert_mask = bert_encoding['attention_mask'].squeeze(0)

        #bert_finput_id = bert_fencoding['input_ids'].squeeze(0)
        #bert_fmask = bert_fencoding['attention_mask'].squeeze(0)


        return {'input_ids':input_ids , 
        'attention_mask': attention_mask , 
        'label': label, 
        'bert_input_id': bert_input_id, 
        'bert_mask': bert_mask 
        }




def create_dataset(is_argument = False ,lecture_path=None, summary_path=None, is_test = False):
    # create datasets
    if is_argument:
        with open('./Datasets/lecture_data_aug.csv',newline='', encoding='utf-8') as f:
          reader = csv.reader(f)
          rows = list(reader)
          rows = rows[1:]
        data_set = list(zip(*rows))
        test_Lecture = list(data_set[1][2480:])
        test_summary = list(data_set[2][2480:])
        t_Lecture = list(data_set[1][:2400])
        t_summary = list(data_set[2][:2400])
        v_Lecture = list(data_set[1][2400:2480])
        v_summary = list(data_set[2][2400:2480])
    else:
        with open('./Datasets/generated_lectures.csv',newline='', encoding='utf-8') as f:
          reader = csv.reader(f)
          rows = list(reader)
          rows = rows[1:]
        data_set = list(zip(*rows))
        test_Lecture = list(data_set[3][900:])
        test_summary = list(data_set[4][900:])
        t_Lecture = list(data_set[3][:800])
        t_summary = list(data_set[4][:800])
        v_Lecture = list(data_set[3][800:900])
        v_summary = list(data_set[4][800:900])
        # create and return datasets
    test_dataset = GANBARTDataset(test_Lecture, test_summary)
    train_dataset = GANBARTDataset(t_Lecture, t_summary)
    validation_dataset = GANBARTDataset(v_Lecture, v_summary)
    del data_set
    del t_Lecture
    del t_summary
    del v_Lecture
    del v_summary
    del test_Lecture
    del test_summary
    return train_dataset, validation_dataset, test_dataset

def samsum_dataset(is_argument = False ,lecture_path=None, summary_path=None, is_test = False):
    # create datasets
    if is_argument:
        with open('/content/drive/MyDrive/COMP_8730_samsum/Datasets/samsum.csv',newline='', encoding='utf-8') as f:
          reader = csv.reader(f)
          rows = list(reader)
          rows = rows[1:]
        data_set = list(zip(*rows))
        test_Lecture = list(data_set[1][2480:])
        test_summary = list(data_set[2][2480:])
        t_Lecture = list(data_set[1][:2400])
        t_summary = list(data_set[2][:2400])
        v_Lecture = list(data_set[1][2400:2480])
        v_summary = list(data_set[2][2400:2480])
    else:
        with open('/content/drive/MyDrive/COMP_8730_samsum/Datasets/samsum.csv',newline='', encoding='utf-8') as f:
          reader = csv.reader(f)
          rows = list(reader)
          rows = rows[1:]
        data_set = list(zip(*rows))
        print(len(data_set[1]))
        test_Lecture = list(data_set[1][4400:4800]) # 12786
        test_summary = list(data_set[2][4400:4800]) # 12786
        t_Lecture = list(data_set[1][:4000]) # 11786
        t_summary = list(data_set[2][:4000]) # 11786
        v_Lecture = list(data_set[1][4000:4400])
        v_summary = list(data_set[2][4000:4400])
        # create and return datasets
    test_dataset = GANBARTDataset(test_Lecture, test_summary)
    train_dataset = GANBARTDataset(t_Lecture, t_summary)
    validation_dataset = GANBARTDataset(v_Lecture, v_summary)
    del data_set
    del t_Lecture
    del t_summary
    del v_Lecture
    del v_summary
    del test_Lecture
    del test_summary
    return train_dataset, validation_dataset, test_dataset

def get_samsum():
    ds = load_dataset("knkarthick/samsum")
    train_dataset = GANBARTDataset(ds["train"]["dialogue"], ds["train"]["summary"])
    validation_dataset = GANBARTDataset(ds["validation"]["dialogue"], ds["validation"]["summary"])
    test_dataset = GANBARTDataset(ds["test"]["dialogue"], ds["test"]["summary"])
    del ds
    return train_dataset, validation_dataset, test_dataset


