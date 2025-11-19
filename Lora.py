from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BartTokenizer, BartModel, BartConfig, AutoConfig, BertForSequenceClassification, set_seed, RobertaTokenizer, RobertaForSequenceClassification
import torch.nn as nn
import torch
from functools import partial

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
set_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B) # delta_W = a*BAW
        return x


class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x) # new_W = W + delta_W


def Lora_fine_tuning_BART(model, lora_r, lora_alpha, lora_dropout, lora_query,
                    lora_key, lora_value, lora_projection, lora_mlp, lora_head
                    ):

    for param in model.parameters():
      param.requires_grad = False # Freeze all parameters in the model BART

    layers = []

    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)
    # for each linear layer add Lora adapter
    for layer in model.model.encoder.layers:
        if lora_query:
            layer.self_attn.q_proj = assign_lora(layer.self_attn.q_proj)
        if lora_key:
            layer.self_attn.k_proj = assign_lora(layer.self_attn.k_proj)
        if lora_value:
            layer.self_attn.v_proj = assign_lora(layer.self_attn.v_proj)
        if lora_projection:
            layer.self_attn.out_proj = assign_lora(layer.self_attn.out_proj)
        if lora_mlp:
            layer.fc1 = assign_lora(layer.fc1)
            layer.fc2 = assign_lora(layer.fc2)

    for layer in model.model.decoder.layers:
        if lora_query:
            layer.self_attn.q_proj = assign_lora(layer.self_attn.q_proj)
        if lora_key:
            layer.self_attn.k_proj = assign_lora(layer.self_attn.k_proj)
        if lora_value:
            layer.self_attn.v_proj = assign_lora(layer.self_attn.v_proj)
        if lora_projection:
            layer.self_attn.out_proj = assign_lora(layer.self_attn.out_proj)
        if lora_mlp:
            layer.fc1 = assign_lora(layer.fc1)
            layer.fc2 = assign_lora(layer.fc2)

    return model


def BART_base_model(ckpt):
    # load the base BART model
    model_ckpt = ckpt
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)

    return base_model

def Lora_fine_tuning_BERT(model):

    for param in model.parameters():
      param.requires_grad = False # Freeze all parameter in BERT

    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_query = True
    lora_key = False
    lora_value = True
    lora_projection = False
    lora_intermediate = False
    lora_BertOut = False
    lora_head = False
    lora_pooler = False
    lora_classifier = True

    layers = []

    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)
    # for each bert linear layer add Lora adapter
    for layer in model.bert.encoder.layer:
        if lora_query:
            layer.attention.self.query = assign_lora(layer.attention.self.query)
        if lora_key:
            layer.attention.self.key = assign_lora(layer.attention.self.key)
        if lora_value:
            layer.attention.self.value = assign_lora(layer.attention.self.value)
        if lora_projection:
            layer.attention.output.dense = assign_lora(layer.attention.output.dense)
        if lora_intermediate:
            layer.intermediate.dense = assign_lora(layer.intermediate.dense)
        if lora_BertOut:
            layer.output.dense = assign_lora(layer.output.dense)
    if lora_pooler:
        model.bert.pooler.dense = assign_lora(model.bert.pooler.dense)
    if lora_classifier:
        model.classifier = assign_lora(model.classifier)

    return model

def BERT_base_model():
    # load the BERT base model
    bert_base = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    return bert_base

def RoBERTa_base_model():
    config = AutoConfig.from_pretrained(
        "roberta-base",
        num_labels=2,
        output_hidden_states=True,
        output_attentions=True
    )
    roberta_base = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        config=config
    )
    return roberta_base

def bart_ids_to_roberta(bart_ids, roberta_tokenizer, max_seq_length=256):
    device = bart_ids.device
    ids = bart_ids.clone()

    if ids.size(0) > max_seq_length:
        ids = ids[:max_seq_length]

    # Pad if shorter than max_seq_length
    pad_length = max_seq_length - ids.size(0)
    if pad_length > 0:
        pad = torch.full((pad_length,), roberta_tokenizer.pad_token_id, device=device)
        ids = torch.cat([ids, pad])

    attention_mask = (ids != roberta_tokenizer.pad_token_id).long()

    return ids, attention_mask

def custom_bart_loss(outputs, labels, token_weights=None):

    # Calculate the standard cross-entropy loss
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # The model outputs logits (not probabilities), so we apply the loss function to logits and labels
    logits = outputs.logits
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

    # If token weights are provided, apply them to the loss
    if token_weights is not None:
        # Flatten weights to match the shape of loss
        token_weights = token_weights.view(-1)
        loss = loss * token_weights

    # Return the mean loss (or sum, depending on your needs)
    return loss.mean()








