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
