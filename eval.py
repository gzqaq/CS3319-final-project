from read_dataset import build_graph

import numpy as np
import os
import pandas as pd
import torch
from absl import app, flags, logging
from sklearn import metrics

os.environ["DGLBACKEND"] = "pytorch"

FLAGS = flags.FLAGS
flags.DEFINE_integer("score", 667, "Checkpoint to restore")
flags.DEFINE_string("save_name", "outputs/debug/gcnsage", "Saved name")

BEST_SCORE = 0


def load_model(rel_list, device):
  model_name = os.path.split(FLAGS.save_name)[-1]
  state_dict_pth = f"{FLAGS.save_name}_model_{FLAGS.score}.pt"
  module = __import__(f"models.{model_name}", fromlist=[""])
  model = module.MyModel(rel_list).to(device)
  model.load_state_dict(torch.load(state_dict_pth))

  return model


def main(_):
  device = torch.device("cpu")
  rel_list = [
      ("author", "ref", "paper"),
      ("paper", "cite", "paper"),
      ("author", "coauthor", "author"),
      ("paper", "beref", "author"),
  ]
  _, _, test_refs, refs_to_pred = build_graph(device)
  logging.info("Graph data loaded")
  model = load_model(rel_list, device)
  logging.info("Model loaded")
  node_embeddings = torch.load(f"{FLAGS.save_name}_emb_{FLAGS.score}.pt")
  logging.info("Node embeddings loaded")

  def prob_fn(feat_u, feat_v):
    with torch.no_grad():
      return model.pred.predict(feat_u, feat_v).cpu().numpy()

  test_arr = np.array(test_refs.values)
  label_true = test_refs.label.to_numpy()

  probs = prob_fn(
      node_embeddings["author"][test_arr[:, 0]],
      node_embeddings["paper"][test_arr[:, 1]],
  )

  preds = np.where(probs > 0.5, 1, 0)
  score = metrics.f1_score(label_true, preds)
  logging.info(f"Test F1-score: {score}")

  test_arr = np.array(refs_to_pred)
  probs = prob_fn(
      node_embeddings["author"][test_arr[:, 0]],
      node_embeddings["paper"][test_arr[:, 1]],
  )
  preds = np.where(probs > 0.5, 1, 0)

  data = []
  for index, p in enumerate(list(preds)):
    tp = [index, str(int(p))]
    data.append(tp)

  df = pd.DataFrame(data, columns=["Index", "Predicted"], dtype=object)
  pth = f"{FLAGS.save_name}_sub_{round(score * 1000)}.csv"
  df.to_csv(pth, index=False)
  logging.info("Save to %s", pth)


if __name__ == "__main__":
  app.run(main)
