from .blocks import GCNBlock, SAGEBlock, GATBlock, MLPPredictor, ScorePredictor

import torch.nn as nn

#######################
# No BatchNorm Blocks #
#######################


class GCNnoBN(GCNBlock):
  def __init__(self, in_feat, hidden_dims, out_feat, rel_names):
    super().__init__(in_feat, hidden_dims, out_feat, rel_names)

    del self.bns

  def forward(self, blocks, x):
    for i, conv in enumerate(self.convs):
      x = conv(blocks[i], x)

    return x


class SAGEnoBN(SAGEBlock):
  def __init__(self, in_feat, hidden_dims, out_feat, rel_names):
    super().__init__(in_feat, hidden_dims, out_feat, rel_names)

    del self.bns

  def forward(self, blocks, x):
    for i, conv in enumerate(self.convs):
      x = conv(blocks[i], x)

    return x


class GATnoBN(GATBlock):
  def __init__(self, in_feat, hidden_dims, out_feat, n_heads, rel_names):
    super().__init__(in_feat, hidden_dims, out_feat, n_heads, rel_names)

    del self.bns

  def forward(self, blocks, x):
    for i, conv in enumerate(self.convs):
      x = conv(blocks[i], x)

    return x


#######################
##### Vanilla NN ######
#######################


class DoubleGCN(nn.Module):
  def __init__(self, rel_names):
    super().__init__()

    out_feat = 64
    self.gcn = GCNBlock(512, [256], out_feat, rel_names)
    self.pred = MLPPredictor(out_feat)

  def gnn(self, g, x):
    return self.gcn(g, x)

  def forward(self, pos_graph, neg_graph, blocks, x):
    x = self.gnn(blocks, x)
    pos_score = self.pred(pos_graph, x)
    neg_score = self.pred(neg_graph, x)

    return pos_score, neg_score

  @property
  def n_layers(self):
    return (2,)


class DoubleSAGE(nn.Module):
  def __init__(self, rel_names):
    super().__init__()

    out_feat = 64
    self.sage = SAGEBlock(512, [256], out_feat, rel_names)
    self.pred = MLPPredictor(out_feat)

  def gnn(self, g, x):
    return self.sage(g, x)

  def forward(self, pos_graph, neg_graph, blocks, x):
    x = self.gnn(blocks, x)
    pos_score = self.pred(pos_graph, x)
    neg_score = self.pred(neg_graph, x)

    return pos_score, neg_score

  @property
  def n_layers(self):
    return (2,)


class DoubleGAT(nn.Module):
  def __init__(self, rel_names):
    super().__init__()

    out_feat = 64
    self.gat = GATBlock(512, [256], out_feat, [4] * 2, rel_names)
    self.pred = MLPPredictor(out_feat * 4)

  def gnn(self, g, x):
    return self.gat(g, x)

  def forward(self, pos_graph, neg_graph, blocks, x):
    x = self.gnn(blocks, x)
    pos_score = self.pred(pos_graph, x)
    neg_score = self.pred(neg_graph, x)

    return pos_score, neg_score

  @property
  def n_layers(self):
    return (2,)


#######################
## Cosine Predictor ###
#######################


class DoubleGCNcos(nn.Module):
  def __init__(self, rel_names):
    super().__init__()

    out_feat = 64
    self.gcn = GCNBlock(512, [256], out_feat, rel_names)
    self.pred = ScorePredictor()

  def gnn(self, g, x):
    return self.gcn(g, x)

  def forward(self, pos_graph, neg_graph, blocks, x):
    x = self.gnn(blocks, x)
    pos_score = self.pred(pos_graph, x)
    neg_score = self.pred(neg_graph, x)

    return pos_score, neg_score

  @property
  def n_layers(self):
    return (2,)


class DoubleSAGEcos(nn.Module):
  def __init__(self, rel_names):
    super().__init__()

    out_feat = 64
    self.sage = SAGEBlock(512, [256], out_feat, rel_names)
    self.pred = ScorePredictor()

  def gnn(self, g, x):
    return self.sage(g, x)

  def forward(self, pos_graph, neg_graph, blocks, x):
    x = self.gnn(blocks, x)
    pos_score = self.pred(pos_graph, x)
    neg_score = self.pred(neg_graph, x)

    return pos_score, neg_score

  @property
  def n_layers(self):
    return (2,)


class DoubleGATcos(nn.Module):
  def __init__(self, rel_names):
    super().__init__()

    out_feat = 64
    self.gat = GATBlock(512, [256], out_feat, [4] * 2, rel_names)
    self.pred = ScorePredictor()

  def gnn(self, g, x):
    return self.gat(g, x)

  def forward(self, pos_graph, neg_graph, blocks, x):
    x = self.gnn(blocks, x)
    pos_score = self.pred(pos_graph, x)
    neg_score = self.pred(neg_graph, x)

    return pos_score, neg_score

  @property
  def n_layers(self):
    return (2,)


#######################
### No BatchNorm NN ###
#######################


class DoubleGCNnoBN(nn.Module):
  def __init__(self, rel_names):
    super().__init__()

    out_feat = 64
    self.gcn = GCNnoBN(512, [256], out_feat, rel_names)
    self.pred = MLPPredictor(out_feat)

  def gnn(self, g, x):
    return self.gcn(g, x)

  def forward(self, pos_graph, neg_graph, blocks, x):
    x = self.gnn(blocks, x)
    pos_score = self.pred(pos_graph, x)
    neg_score = self.pred(neg_graph, x)

    return pos_score, neg_score

  @property
  def n_layers(self):
    return (2,)


class DoubleSAGEnoBN(nn.Module):
  def __init__(self, rel_names):
    super().__init__()

    out_feat = 64
    self.sage = SAGEnoBN(512, [256], out_feat, rel_names)
    self.pred = MLPPredictor(out_feat)

  def gnn(self, g, x):
    return self.sage(g, x)

  def forward(self, pos_graph, neg_graph, blocks, x):
    x = self.gnn(blocks, x)
    pos_score = self.pred(pos_graph, x)
    neg_score = self.pred(neg_graph, x)

    return pos_score, neg_score

  @property
  def n_layers(self):
    return (2,)


class DoubleGATnoBN(nn.Module):
  def __init__(self, rel_names):
    super().__init__()

    out_feat = 64
    self.gat = GATnoBN(512, [256], out_feat, [4] * 2, rel_names)
    self.pred = MLPPredictor(out_feat * 4)

  def gnn(self, g, x):
    return self.gat(g, x)

  def forward(self, pos_graph, neg_graph, blocks, x):
    x = self.gnn(blocks, x)
    pos_score = self.pred(pos_graph, x)
    neg_score = self.pred(neg_graph, x)

    return pos_score, neg_score

  @property
  def n_layers(self):
    return (2,)
