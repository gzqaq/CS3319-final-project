from models import max_margin_loss
from read_dataset import build_graph

import dgl
import numpy as np
import os
import time
import torch
from absl import app, flags, logging
from sklearn import metrics

os.environ["DGLBACKEND"] = "pytorch"

FLAGS = flags.FLAGS
flags.DEFINE_integer("n_epochs", 5, "Number of epochs")
flags.DEFINE_float("lr", 0.001, "Learning rate")
flags.DEFINE_string("ckpt", "", "Checkpoint to restore")
flags.DEFINE_string("gpu", "0", "Which GPU to use")
flags.DEFINE_string("model", "gcnsage", "Model type (must defined in models)")
flags.DEFINE_string("run_name", "debug", "Run name")

BEST_SCORE = 0


def build_model(rel_list, device):
  module = __import__(f"models.{FLAGS.model}", fromlist=[""])

  return module.MyModel(rel_list).to(device)


def eval_and_save_checkpoint(device,
                             model,
                             node_features,
                             g,
                             test_refs,
                             out_dir,
                             last: bool = False):
  global BEST_SCORE
  model.eval()

  node_features = {k: v.to(device) for k, v in node_features.items()}
  with torch.no_grad():
    node_embeddings = model.gnn([g] * sum(model.n_layers), node_features)
  del node_features
  torch.cuda.empty_cache()

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
  logging.info("Test f1-score: %f", score)

  if score > BEST_SCORE or last:
    BEST_SCORE = score
    score_int = round(1000 * score)

    emb_pth = f"{out_dir}/{FLAGS.model}_emb_{score_int}.pt"
    state_dict_pth = f"{out_dir}/{FLAGS.model}_model_{score_int}.pt"

    node_embeddings = {k: v.cpu() for k, v in node_embeddings.items()}
    torch.save(node_embeddings, emb_pth)
    logging.info("Save node embeddings to %s", emb_pth)

    torch.save(model.state_dict(), state_dict_pth)
    logging.info("Save model weights to %s", state_dict_pth)


def main(_):
  device = torch.device(f"cuda:{FLAGS.gpu}")
  out_dir = os.path.join("outputs", FLAGS.run_name)
  os.makedirs(out_dir, exist_ok=True)

  rel_list = [
      ("author", "ref", "paper"),
      ("paper", "cite", "paper"),
      ("author", "coauthor", "author"),
      ("paper", "beref", "author"),
  ]
  node_features, g, test_refs, _ = build_graph(device)
  logging.info("Graph data loaded")
  model = build_model(rel_list, device)
  logging.info("Model built")

  sampler = dgl.dataloading.MultiLayerFullNeighborSampler(sum(model.n_layers))
  # sampler = dgl.dataloading.NeighborSampler([4, 4])
  train_eid_dict = {
      etype: g.edges(etype=etype, form="eid") for etype in g.etypes
  }
  sampler = dgl.dataloading.as_edge_prediction_sampler(
      sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(5))
  dataloader = dgl.dataloading.DataLoader(
      g,
      train_eid_dict,
      sampler,
      device=device,
      batch_size=32768,
      shuffle=True,
      drop_last=False,
      num_workers=0,
  )

  opt = torch.optim.Adam(model.parameters(), FLAGS.lr)
  scheduler = torch.optim.lr_scheduler.ChainedScheduler([
      torch.optim.lr_scheduler.ConstantLR(opt, 0.9, 10),
      torch.optim.lr_scheduler.CosineAnnealingLR(opt, FLAGS.n_epochs),
  ])

  if FLAGS.ckpt != "" and os.path.exists(FLAGS.ckpt):
    model.load_state_dict(torch.load(FLAGS.ckpt))

  model.train()
  logging.info("Start training...")
  try:
    eval_between = 0
    for i_epoch in range(FLAGS.n_epochs):
      avg_loss = 0
      beg = time.time()
      for step, (_, pos_g, neg_g, blocks) in enumerate(dataloader):
        input_features = blocks[0].srcdata["features"]
        pos_score, neg_score = model(pos_g, neg_g, blocks, input_features)
        loss = max_margin_loss(pos_score, neg_score, rel_list[0])

        opt.zero_grad()
        loss.backward()
        opt.step()

        avg_loss += (loss.item() - avg_loss) / (step + 1)

      logging.info(f"| Epoch {i_epoch:3d} "
                   f"| time: {time.time() - beg:<7.3f} "
                   f"| loss: {avg_loss:.3f} |")

      if avg_loss < 0.15 and eval_between > 2:
        eval_and_save_checkpoint(device, model, node_features, g, test_refs,
                                 out_dir)
        model.train()
        eval_between = 0

      scheduler.step()
      eval_between += 1

  except KeyboardInterrupt:
    pass

  eval_and_save_checkpoint(device,
                           model,
                           node_features,
                           g,
                           test_refs,
                           out_dir,
                           last=True)


if __name__ == "__main__":
  app.run(main)
