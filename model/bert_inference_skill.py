import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Data
def load_data(PATH):
    small_df = pd.read_pickle(PATH)
    labels = list(small_df["label"])
    polarity = [0 if l == "positive" else 1 for l in labels]
    small_df["polarity"] = polarity
    return small_df

# Load Model
def load_model(PATH):
    config = BertConfig.from_pretrained(PATH + "/config.json", output_hidden_states=True)
    bert_model = BertForSequenceClassification.from_pretrained(PATH, config=config)
    bert_model.to(device)

    BERTMODEL = "dbmdz/bert-base-italian-cased"
    tokenizer = BertTokenizer.from_pretrained(BERTMODEL, do_lower_case=True)
    print("Model loaded...")
    return bert_model, tokenizer

# Process Dataframe
def process_dataframe(_dframe, _tokenizer, batch_size):
    sentences = _dframe.text.values
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(_dframe.label)
    tokenized = [_tokenizer.encode(s, add_special_tokens=True, max_length=MAX_LEN_TRAIN, truncation=True) for s in sentences]
    ids = np.array([np.pad(i, (0, MAX_LEN_TRAIN - len(i)), mode='constant') for i in tokenized])
    amasks = np.array([[float(i > 0) for i in seq] for seq in ids])

    inputs_reformatted = torch.tensor(ids)
    labels_reformatted = torch.tensor(labels)
    masks_reformatted = torch.tensor(amasks)

    data = TensorDataset(inputs_reformatted, masks_reformatted, labels_reformatted)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader

# Run Model
def run_model(_model, loader):
    ce_loss = nn.CrossEntropyLoss()
    _model.eval()
    all_losses = []

    for step, batch in enumerate(loader):
        b_input_ids = batch[0].to(device, dtype=torch.long)
        b_input_mask = batch[1].to(device, dtype=torch.float)
        b_labels = batch[2].to(device, dtype=torch.long)

        with torch.no_grad():
            outputs = _model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs.logits

        loss_val_list = ce_loss(logits, b_labels)
        all_losses.append(loss_val_list.item())
    mean_loss = np.mean(all_losses)
    print("Inference loss:", mean_loss)

# Get Sentence Activation
def get_sentence_activation(DATAPATH, MODELPATH, batch_size):
    sentence_df = pd.read_pickle(DATAPATH)
    model, tokenizer = load_model(MODELPATH)
    loader = process_dataframe(sentence_df, tokenizer, batch_size)

    extracted_activations = []

    def extract_activation_hook(model, input, output):
        extracted_activations.append(output.cpu().numpy())

    def add_activation_hook(model, layer_idx):
        all_modules_list = list(model.modules())
        module = all_modules_list[layer_idx]
        module.register_forward_hook(extract_activation_hook)

    add_activation_hook(model, layer_idx=-2)
    print("Running inference...")
    run_model(model, loader)

    return np.concatenate(extracted_activations, axis=0)

# Save Activations
def save_activations(activations, DATAPATH):
    np.save(DATAPATH, activations)
