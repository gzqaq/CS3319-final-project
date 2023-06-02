import dgl.function as fn
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticTwoLayerRGCN(nn.Module):
  def __init__(self, in_feat, hidden_feat, out_feat, rel_names):
    super().__init__()

    self.conv1 = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(in_feat, hidden_feat, norm="right") for rel in rel_names})
    self.conv2 = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(hidden_feat, out_feat, norm="right") for rel in rel_names})

  def forward(self, blocks, x):
    x = self.conv1(blocks[0], x)
    x = self.conv2(blocks[1], x)

    return x


class GCNBlock(nn.Module):
  def __init__(self, in_feat, hidden_dims, out_feat, rel_names):
    super().__init__()

    in_feats = [in_feat] + hidden_dims
    out_size = hidden_dims + [out_feat]

    self.convs = nn.ModuleList()
    for in_f, out_f in zip(in_feats, out_size):
      self.convs.append(dglnn.HeteroGraphConv(
        {rel: dglnn.GraphConv(in_f, out_f, norm="right", activation=F.elu) for rel in rel_names}
      ))
  
  def forward(self, blocks, x):
    for i, conv in enumerate(self.convs):
      x = conv(blocks[i], x)

    return x


class SAGEBlock(nn.Module):
  def __init__(self, in_feat, hidden_dims, out_feat, rel_names):
    super().__init__()

    in_feats = [in_feat] + hidden_dims
    out_size = hidden_dims + [out_feat]

    self.convs = nn.ModuleList()
    for in_f, out_f in zip(in_feats, out_size):
      self.convs.append(dglnn.HeteroGraphConv(
        {rel: dglnn.SAGEConv(in_f, out_f, "mean", activation=F.elu) for rel in rel_names}
      ))
  
  def forward(self, blocks, x):
    for i, conv in enumerate(self.convs):
      x = conv(blocks[i], x)

    return x


class GATBlock(nn.Module):
  def __init__(self, in_feat, hidden_dims, out_feat, n_heads, rel_names):
    assert len(hidden_dims) == len(n_heads) - 1
    super().__init__()

    in_feats = [in_feat] + hidden_dims
    out_size = hidden_dims + [out_feat]
    n_heads = [1] + n_heads

    self.convs = nn.ModuleList()
    for i, (in_f, out_f) in enumerate(zip(in_feats, out_size)):
      self.convs.append(dglnn.HeteroGraphConv(
        {rel: dglnn.GATConv(in_f * n_heads[i], out_f, n_heads[i + 1], activation=F.elu) for rel in rel_names}
      ))
  
  def forward(self, blocks, x):
    for i, conv in enumerate(self.convs):
      x = conv(blocks[i], x)
      x = {k: v.flatten(1) for k, v in x.items()}

    return x


class ScorePredictor(nn.Module):
  def forward(self, edge_subgraph, h):
    with edge_subgraph.local_scope():
      edge_subgraph.ndata["h"] = h
      
      for etype in edge_subgraph.canonical_etypes:
        edge_subgraph.apply_edges(fn.u_dot_v("h", "h", "score"), etype=etype)

      return edge_subgraph.edata["score"]


class MLPPredictor(nn.Module):
  def __init__(self, emb_dim: int):
    super().__init__()

    self.hidden = nn.Linear(emb_dim * 2, 2048)
    self.out_lyr = nn.Linear(2048, 1)

    self.reset_parameters()

  def reset_parameters(self):
    gain_relu = nn.init.calculate_gain("relu")
    gain_sigmoid = nn.init.calculate_gain("sigmoid")
    nn.init.xavier_uniform_(self.hidden.weight, gain=gain_relu)
    nn.init.xavier_uniform_(self.out_lyr.weight, gain=gain_sigmoid)

  def predict(self, emb_1, emb_2):
    cat_emb = torch.cat([emb_1, emb_2], dim=1)
    x = F.relu(self.hidden(cat_emb))
    ratings = F.sigmoid(self.out_lyr(x))

    return ratings

  def forward(self, g, h):
    rating_dict = {}
    for etype in g.canonical_etypes:
      utype, _, vtype = etype
      src_nid, dst_nid = g.all_edges(etype=etype)
      emb_1 = h[utype][src_nid]
      emb_2 = h[vtype][dst_nid]

      ratings = self.predict(emb_1, emb_2)

      rating_dict[etype] = torch.flatten(ratings)

    rating_dict = {k: torch.unsqueeze(v, dim=1) for k, v in rating_dict.items()}
    return rating_dict
