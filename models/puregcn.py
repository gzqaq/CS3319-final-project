from .blocks import GCNBlock, MLPPredictor

import torch.nn as nn


class MyModel(nn.Module):
  def __init__(self, rel_names):
    super().__init__()

    out_feat = 64
    self.gcn = GCNBlock(512, [512, 256, 128], out_feat, rel_names)
    self.pred = MLPPredictor(out_feat)

  def gnn(self, g, x):
    x = self.gcn(g, x)

    return x

  def forward(self, pos_graph, neg_graph, blocks, x):
    x = self.gnn(blocks, x)
    pos_score = self.pred(pos_graph, x)
    neg_score = self.pred(neg_graph, x)

    return pos_score, neg_score

  @property
  def n_layers(self):
    return (4,)
