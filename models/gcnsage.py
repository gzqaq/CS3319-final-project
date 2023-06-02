from .blocks import GCNBlock, SAGEBlock, MLPPredictor

import torch.nn as nn


class MyModel(nn.Module):
  def __init__(self, rel_names):
    super().__init__()

    out_feat = 32
    self.gcn = GCNBlock(512, [1024, 256], 128, rel_names)
    self.save = SAGEBlock(128, [64, 32], out_feat, rel_names)
    self.pred = MLPPredictor(out_feat * 4)

  def gnn(self, g, x):
    x = self.gcn(g[:self.n_layers[0]], x)
    x = self.sage(g[self.n_layers[0]:], x)

    return x

  def forward(self, pos_graph, neg_graph, blocks, x):
    x = self.gnn(blocks, x)
    pos_score = self.pred(pos_graph, x)
    neg_score = self.pred(neg_graph, x)

    return pos_score, neg_score

  @property
  def n_layers(self):
    return (3, 3)
