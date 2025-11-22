import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import RobertaModel, RobertaConfig

from transformers import AutoModelForSeq2SeqLM

def emb_triplet_loss(positives, negatives, anchors):
  margin = 0.3
  anchor = F.normalize(anchors, p=2, dim=1)
  negatives = F.normalize(negatives, p=2, dim=1)
  positives = F.normalize(positives, p=2, dim=1)

  pos_dist = F.cosine_similarity(anchor, positives, dim=1)
  neg_dist = F.cosine_similarity(anchor, negatives, dim=1)
  #print(pos_dist.mean().item(), neg_dist.mean().item())
  loss = torch.relu(pos_dist - neg_dist + margin).mean()
  return loss

def classification_loss(logits, labels):
  criterion = nn.BCEWithLogitsLoss()
  loss = criterion(logits, labels)
  return loss

def BART_base_model(ckpt):
    # load the base BART model
    model_ckpt = 'facebook/bart-base'
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)

    return base_model


class RobertaNoEmbedding(RobertaModel):
    def __init__(self, config):
        super().__init__(config)
        # disable token / position embeddings
        # Override embeddings forward to just return inputs_embeds
        self.embeddings.forward = lambda inputs_embeds=None, **kwargs: inputs_embeds

    def forward(self, inputs_embeds, attention_mask=None, **kwargs):
        # Make sure input_ids are None
        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            token_type_ids=None,
            **kwargs
        )


class Roberta_discriminator(nn.Module):
    def __init__(self, num_labels, pretrained_name="roberta-base"):
        super().__init__()
        config = RobertaConfig.from_pretrained(pretrained_name)
        self.roberta_noemb = RobertaNoEmbedding(config)
        self.classifier = nn.Linear(self.roberta_noemb.config.hidden_size, num_labels)

    def forward(self, x_emb, attention_mask=None):
        outputs = self.roberta_noemb(inputs_embeds=x_emb, attention_mask=attention_mask)
        logits = self.classifier(outputs["pooler_output"])
        return logits