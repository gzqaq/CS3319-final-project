from .blocks import GATBlock, MLPPredictor

import torch.nn as nn


class MyModel(nn.Module):
  def __init__(self, rel_names):
    super().__init__()

    out_feat = 32
    self.gat = GATBlock(512, [256, 256, 128, 128, 64, 64], out_feat, [4] * 7, rel_names)
    self.pred = MLPPredictor(out_feat * 4)
  
  def gnn(self, g, x):
    return self.gat(g[:self.n_layers[0]], x)
  
  def forward(self, pos_graph, neg_graph, blocks, x):
    x = self.gnn(blocks, x)
    pos_score = self.pred(pos_graph, x)
    neg_score = self.pred(neg_graph, x)

    return pos_score, neg_score
  
  @property
  def n_layers(self):
    return (7,)