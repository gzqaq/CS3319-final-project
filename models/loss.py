import torch
import torch.nn.functional as F


def max_margin_loss(pos_score,
                    neg_score,
                    etype,
                    neg_sample_size=5,
                    delta: float = 0.9):
  neg_score_tensor = neg_score[etype]
  pos_score_tensor = pos_score[etype]
  neg_score_tensor = neg_score_tensor.reshape(-1, neg_sample_size)
  scores = neg_score_tensor + delta - pos_score_tensor

  return F.relu(scores).mean()


def binary_cross_entropy_loss(pos_score, neg_score, etype):
  score = torch.cat([pos_score[etype], neg_score[etype]])
  label = torch.cat([torch.ones_like(pos_score[etype]), torch.zeros_like(neg_score[etype])])

  return F.binary_cross_entropy_with_logits(score, label)