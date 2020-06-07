import torch
from conceptNet import ConceptNet
import numpy as np
import pandas as pd
from transformers import BertForSequenceClassification
from tqdm import tqdm
import argparse
from tensorboardX import SummaryWriter
from pathlib import Path
import matplotlib.pyplot as plt

from interpretConcepts import plot_embeddings, save_concepts, concept_analysis

def train(args, train_embeddings, train_y_true, h_x, n_concepts, writer, device):
  '''
  :param train_embeddings: tensor of sentence embeddings => (# of examples, embedding_dim)
  :param train_y_true: the ground truth label for each of the embeddings => (# of examples)
  :param clusters: tensor of embedding clusters generated by k-means => (# of n_clusters, # of sentences per cluster, embedding_dim)
  :param h_x: final layers of the transformer
  :param n_concepts: number of concepts to generate
  :return: trained conceptModel
  '''

  # training parameters
  l_1 = args.l1
  l_2 = args.l2
  topk = args.topk
  batch_size = args.batch_size
  epochs = args.num_epochs
  cal_interval = args.shapley_interval
  train_embeddings = torch.from_numpy(train_embeddings).to(device)
  train_y_true = torch.from_numpy(train_y_true.astype('int64')).to(device)

  for p in list(h_x.parameters()):
    p.requires_grad = False

  model = ConceptNet(n_concepts, train_embeddings).to(device)

  save_dir = Path(args.save_dir)
  save_dir.mkdir(exist_ok=True, parents=True)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  train_size = train_embeddings.shape[0]
  loss_reg_epoch = args.loss_reg_epoch
  losses = []

  n_iter = 0
  for i in tqdm(range(epochs)):
    if i < loss_reg_epoch:
      regularize = False
    else:
      regularize = True

    batch_start = 0
    batch_end = batch_size

    # do a shuffle of train_embeddings, train_y_true
    train_y_true_float = train_y_true.float().unsqueeze(dim=1)
    data_pair = torch.cat([train_embeddings, train_y_true_float], dim=1)
    new_permute = torch.randperm(data_pair.shape[0])
    data_pair = data_pair[new_permute]
    permuted_train_embeddings = data_pair[:, :-1]
    permuted_train_y_true = data_pair[:, -1].long()

    while batch_end < train_size:
      # generate training batch
      train_embeddings_narrow = permuted_train_embeddings.narrow(0, batch_start, batch_end - batch_start)
      train_y_true_narrow = permuted_train_y_true.narrow(0, batch_start, batch_end - batch_start)
      if (n_iter) % cal_interval == 0:
        completeness, conceptSHAP, final_loss, pred_loss, l1, l2, metrics = model.loss(train_embeddings_narrow,
                                                                                       train_y_true_narrow, h_x,
                                                                                       regularize=regularize,
                                                                                       doConceptSHAP=True,
                                                                                       l_1=l_1, l_2=l_2, topk=topk)
      else:
        completeness, conceptSHAP, final_loss, pred_loss, l1, l2, metrics = model.loss(train_embeddings_narrow,
                                                                                       train_y_true_narrow, h_x,
                                                                                       regularize=regularize,
                                                                                       doConceptSHAP=False,
                                                                                       l_1=l_1, l_2=l_2, topk=topk)
      # update gradients
      optimizer.zero_grad()
      final_loss.backward()
      optimizer.step()

      # logging
      writer.add_scalar('sum_loss', final_loss.data.item(), n_iter)
      writer.add_scalar('pred_loss', pred_loss.data.item(), n_iter)
      writer.add_scalar('L1', l1.data.item(), n_iter)
      writer.add_scalar('L2', l2.data.item(), n_iter)
      writer.add_scalar('norm_metrics', metrics[0].data.item(), n_iter)
      writer.add_scalar('concept completeness', completeness.data.item(), n_iter)
      if conceptSHAP != []:
        fig = plt.figure()
        plt.bar(list(range(len(conceptSHAP))), conceptSHAP)
        writer.add_figure('conceptSHAP', fig, n_iter)

      # update batch indices
      batch_start += batch_size
      batch_end += batch_size
      n_iter += 1

  return model, losses


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # Required dependencies
  parser.add_argument("--activation_dir", type=str, required=True,
                      help="path to .npy file containing dataset embeddings")
  parser.add_argument("--train_dir", type=str, required=True,
                      help="path to .pkl file containing train dataset")
  parser.add_argument("--bert_weights", type=str, required=True,
                      help="path to BERT config & weights directory")
  parser.add_argument("--n_concepts", type=int, default=5,
                      help="number of concepts to generate")

  # Training options
  parser.add_argument('--save_dir', default='./experiments',
                      help='directory to save the model')
  parser.add_argument('--log_dir', default='./logs',
                      help='directory to save the log')
  parser.add_argument('--l1', type=float, default=.001)
  parser.add_argument('--l2', type=float, default=.002)
  parser.add_argument('--topk', type=int, default=10)
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--loss_reg_epoch', type=int, default=2,
                      help="num of epochs to run without loss regularization")
  parser.add_argument('--num_epochs', type=int, default=3,
                      help="num of training epochs")
  parser.add_argument('--shapley_interval', type=int, default=5)
  args = parser.parse_args()

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # init tensorboard
  writer = SummaryWriter()

  ###############################
  # Preparing data
  ###############################
  print("Loading dataset embeddings...")
  small_activations = np.load(args.activation_dir)
  print("Shape: " + str(small_activations.shape))

  print("Loading dataset labels...")
  data_frame = pd.read_pickle(args.train_dir)
  senti_list = np.array(data_frame['polarity'])

  print("Loading model weights...")
  bert_model = BertForSequenceClassification.from_pretrained(args.bert_weights) # ../model/imdb_weights
  bert_model.to(device)  # move to gpu

  print("Init training...\n")
  # get the embedding numpy array, convert to tensor
  train_embeddings = small_activations  # (4012, 768)

  # get ground truth label
  train_y_true = senti_list

  # h_x
  h_x = list(bert_model.modules())[-1]

  # n_concepts
  n_concepts = args.n_concepts  # param

  ###############################
  # Training model
  ###############################
  # init training
  concept_model, loss = train(args, train_embeddings, train_y_true, h_x, n_concepts, writer, device)

  ###############################
  # Interpretation of results
  ###############################

  # save concepts
  save_concepts(concept_model)

  # plot activations / clusters / concepts
  plot_embeddings(train_embeddings, data_frame, senti_list, writer)

  # eval concepts
  concept_analysis(small_activations, data_frame)

  writer.close()
