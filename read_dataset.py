import dgl
import os
import pandas as pd
import pickle
import torch
from multiprocessing import cpu_count, Pool
from typing import List, Tuple


def _parse_line(line: str) -> List[int]:
  return list(map(int, line.strip().split(" ")))


def read_txt(path: str) -> List[List[int]]:
  with open(path, "r") as fd:
    lines = fd.readlines()

  with Pool(cpu_count()) as p:
    res = p.map(_parse_line, lines)

  return res


def build_dataset(cite_edges: List[List[int]], ref_edges: List[List[int]], coauthor_edges: List[List[int]], split: float = 0.9, random_state: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
  cite_edges, ref_edges, coauthor_edges = tuple(map(lambda x: pd.DataFrame(x, columns=["source", "target"]), [cite_edges, ref_edges, coauthor_edges]))

  tmp = pd.concat([cite_edges.loc[:, "source"], cite_edges.loc[:, "target"], ref_edges.loc[:, "target"]])
  node_papers = pd.DataFrame(index=pd.unique(tmp))
  tmp = pd.concat([ref_edges["source"], coauthor_edges["source"], coauthor_edges["target"]])
  node_authors = pd.DataFrame(index=pd.unique(tmp))

  print(f"Number of paper nodes: {len(node_papers)}\n"
        f"Number of author nodes: {len(node_authors)}")

  train_refs = ref_edges.sample(frac=split, random_state=random_state)
  test_true_refs = ref_edges[~ref_edges.index.isin(train_refs.index)]
  test_true_refs.loc[:, "label"] = 1

  false_source = node_authors.sample(frac=test_true_refs.shape[0] / node_authors.shape[0],
                                     random_state=random_state,
                                     replace=True, axis=0).reset_index()
  false_target = node_papers.sample(frac=test_true_refs.shape[0] / node_papers.shape[0],
                                     random_state=random_state,
                                     replace=True, axis=0).reset_index()
  test_false_refs = pd.concat([false_source, false_target], axis=1)
  test_false_refs.columns = ["source", "target"]
  test_false_refs = test_false_refs[test_false_refs.isin(ref_edges) == False]
  test_false_refs.loc[:, "label"] = 0

  test_refs = pd.concat([test_true_refs, test_false_refs.iloc[:min(len(false_source), len(false_target))]])
  test_refs = test_refs.sample(frac=1, random_state=random_state, axis=0)

  return train_refs, test_refs


def build_graph(device):
  ds_dir = os.path.abspath("dataset")   # Directory that stores following files
  cite_file = "paper_file_ann.txt"
  train_ref_file = "bipartite_train_ann.txt"
  test_ref_file = "bipartite_test_ann.txt"
  coauthor_file = "author_file_ann.txt"
  feature_file = "feature.pkl"

  citation = read_txt(os.path.join(ds_dir, cite_file))
  existing_refs = read_txt(os.path.join(ds_dir, train_ref_file))
  refs_to_pred = read_txt(os.path.join(ds_dir, test_ref_file))
  coauthor = read_txt(os.path.join(ds_dir, coauthor_file))

  with open(os.path.join(ds_dir, feature_file), "rb") as fd:
    paper_features = pickle.load(fd)

  train_refs, test_refs, cite_edges, coauthor_edges = build_dataset(citation, existing_refs, coauthor)
  train_ref_tensor = torch.from_numpy(train_refs.values)
  cite_tensor = torch.from_numpy(cite_edges.values)
  coauthor_tensor = torch.from_numpy(coauthor_edges.values)

  rel_list = [("author", "ref", "paper"), ("paper", "cite", "paper"), ("author", "coauthor", "author"), ("paper", "beref", "author")]
  graph_data = {
      rel_list[0]: (train_ref_tensor[:, 0], train_ref_tensor[:, 1]),
      rel_list[1]: (torch.cat([cite_tensor[:, 0], cite_tensor[:, 1]]), torch.cat([cite_tensor[:, 1], cite_tensor[:, 0]])),
      rel_list[2]: (torch.cat([coauthor_tensor[:, 0], coauthor_tensor[:, 1]]), torch.cat([coauthor_tensor[:, 1], coauthor_tensor[:, 0]])),
      rel_list[3]: (train_ref_tensor[:, 1], train_ref_tensor[:, 0]),
  }

  hetero_graph = dgl.heterograph(graph_data)
  node_features = {'author': torch.rand(6611, 512), 'paper': paper_features}
  hetero_graph.ndata['features'] = node_features
  hetero_graph = hetero_graph.int().to(device)

  return node_features, hetero_graph, test_refs, refs_to_pred