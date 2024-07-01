import os
import random
import json
import argparse
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset(data_dir, tokenizer, max_seq_length):
    # Function to load datasets
    # (Implementation remains unchanged)

def set_seed(seed):
    # Function to set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

def train(train_dataset, model, tokenizer, args):
    # Training function
    # (Implementation remains unchanged)

def evaluate(eval_dataset, model, tokenizer, args):
    # Evaluation function
    # (Implementation remains unchanged)

def run_model(_model, loader):
    ce_loss = nn.CrossEntropyLoss()
    _model.eval()
    all_losses = []

    for step, batch in enumerate(loader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            outputs = _model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs.logits

        print(f"logits type: {type(logits)}")
        print(f"b_labels type: {type(b_labels)}")

        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits).to(device)

        if not isinstance(b_labels, torch.Tensor):
            b_labels = torch.tensor(b_labels).to(device)

        loss_val_list = ce_loss(logits, b_labels)
        all_losses.append(loss_val_list.item())

    mean_loss = np.mean(all_losses)
    print("inference loss:", mean_loss)

    all_losses = []
    for batch in tqdm(loader):
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_input_mask = b_input_mask.to(device)
        b_labels = b_labels.to(device)
        with torch.no_grad():
            outputs = _model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs.logits
            loss_val_list = ce_loss(logits, b_labels)
            pred_loss = torch.mean(loss_val_list).item()
            all_losses.append(pred_loss)
    print("inference loss:", np.mean(np.array(all_losses)))

def main():
    parser = argparse.ArgumentParser()
    # Add argument parsing here
    # (Implementation remains unchanged)

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    config = BertConfig.from_pretrained(args.bert_weights, num_labels=args.num_labels)
    tokenizer = BertTokenizer.from_pretrained(args.bert_weights)
    model = BertModel.from_pretrained(args.bert_weights, config=config)

    model.to(device)

    # Load data
    train_dataset = load_dataset(args.train_dir, tokenizer, args.max_seq_length)
    eval_dataset = load_dataset(args.eval_dir, tokenizer, args.max_seq_length)

    if args.do_train:
        train(train_dataset, model, tokenizer, args)

    if args.do_eval:
        evaluate(eval_dataset, model, tokenizer, args)

if __name__ == "__main__":
    main()
