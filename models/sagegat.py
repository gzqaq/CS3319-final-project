from .blocks import SAGEBlock, GATBlock, MLPPredictor

import torch.nn as nn


class MyModel(nn.Module):
  def __init__(self, rel_names):
    super().__init__()

    out_feat = 64
    self.sage = SAGEBlock(512, [1024, 512], 256, rel_names)
    self.gat = GATBlock(256, [512, 256, 128], out_feat, [4] * 4, rel_names)
    self.pred = MLPPredictor(out_feat * 4)

  def gnn(self, g, x):
    x = self.sage(g[:self.n_layers[0]], x)
    x = self.gat(g[self.n_layers[0]:], x)

    return x

  def forward(self, pos_graph, neg_graph, blocks, x):
    x = self.gnn(blocks, x)
    pos_score = self.pred(pos_graph, x)
    neg_score = self.pred(neg_graph, x)

    return pos_score, neg_score

  @property
  def n_layers(self):
    return (3, 4)
