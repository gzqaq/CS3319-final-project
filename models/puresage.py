from .blocks import SAGEBlock, MLPPredictor

import torch.nn as nn


class MyModel(nn.Module):
  def __init__(self, rel_names):
    super().__init__()

    out_feat = 32
    self.sage = SAGEBlock(512, [1024, 512, 256, 128, 64], out_feat, rel_names)
    self.pred = MLPPredictor(out_feat)

  def gnn(self, g, x):
    x = self.sage(g, x)

    return x

  def forward(self, pos_graph, neg_graph, blocks, x):
    x = self.gnn(blocks, x)
    pos_score = self.pred(pos_graph, x)
    neg_score = self.pred(neg_graph, x)

    return pos_score, neg_score

  @property
  def n_layers(self):
    return (6,)
