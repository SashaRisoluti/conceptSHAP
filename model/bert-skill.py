import torch
import pandas as pd
import numpy as np

from torch.utils.data import (TensorDataset, DataLoader, RandomSampler, SequentialSampler)
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from packaging import version
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import argparse
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import IPython

def train(epoch, loss_vector=None, log_interval=200):
    model.train()
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        if loss_vector is not None:
            loss_vector.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % log_interval == 0:
            print(f'Train Epoch: {epoch} [{step * len(b_input_ids)}/{len(train_dataloader.dataset)} ({100. * step / len(train_dataloader):.0f}%)]\tLoss: {loss:.6f}')

def evaluate(loader):
    model.eval()
    n_correct, n_all = 0, 0
    for batch in loader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        predictions = np.argmax(logits, axis=1)
        labels = b_labels.to('cpu').numpy()
        n_correct += np.sum(predictions == labels)
        n_all += len(labels)
    print(f'Accuracy: [{n_correct}/{n_all}] {n_correct/n_all:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="path to save the model to")
    parser.add_argument("--train_data", type=str, required=True, help="path to the training data for the entire BERT model")
    parser.add_argument("--test_data", type=str, required=True, help="path to the test data for BERT")
    args = parser.parse_args()
    ckpt_path = args.model_dir
    train_path = args.train_data
    test_path = args.test_data

    sns.set()
    matplotlib.use('Agg')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    devicename = f'[{torch.cuda.get_device_name(0)}]' if torch.cuda.is_available() else ""

    print(f'Using PyTorch version: {torch.__version__} Device: {device} {devicename}')
    assert version.parse(torch.__version__) >= version.parse("1.0.0")

    train_df = pd.read_pickle(train_path)
    test_df = pd.read_pickle(test_path)

    print('\nSKILL data loaded:')
    print(f'train: {train_df.shape}')
    print(f'test: {test_df.shape}')

    train_df.sample(10)

    sentences_train = train_df.sentence.values
    sentences_train = ["[CLS] " + s for s in sentences_train]

    sentences_test = test_df.sentence.values
    sentences_test = ["[CLS] " + s for s in sentences_test]

    # Convertiamo i label da float a int
    #train_df.label = train_df.label.apply(lambda x: x*10).values.astype(int)
    labels_train = train_df.label.values
    #test_df.label = test_df.label.apply(lambda x: x*10).values.astype(int)
    labels_test = test_df.label.values

    label_encoder = LabelEncoder()
    labels_train = label_encoder.fit_transform(labels_train)
    labels_test = label_encoder.transform(labels_test)

    print("\nThe first training sentence:")
    print(sentences_train[0], 'LABEL:', labels_train[0])

    # Use the Italian BERT model and Tokenizer
    BERTMODEL = "dbmdz/bert-base-italian-cased"

    tokenizer = BertTokenizer.from_pretrained(BERTMODEL, do_lower_case=True)

    tokenized_train = [tokenizer.tokenize(s) for s in sentences_train]
    tokenized_test = [tokenizer.tokenize(s) for s in sentences_test]

    print("\nThe full tokenized first training sentence:")
    print(tokenized_train[0])

    MAX_LEN_TRAIN, MAX_LEN_TEST = 128, 512

    tokenized_train = [t[:(MAX_LEN_TRAIN - 1)] + ['[SEP]'] for t in tokenized_train]
    tokenized_test = [t[:(MAX_LEN_TEST - 1)] + ['[SEP]'] for t in tokenized_test]

    print("\nThe truncated tokenized first training sentence:")
    print(tokenized_train[0])

    ids_train = [tokenizer.convert_tokens_to_ids(t) for t in tokenized_train]
    ids_train = np.array([np.pad(i, (0, MAX_LEN_TRAIN - len(i)), mode='constant') for i in ids_train])

    ids_test = [tokenizer.convert_tokens_to_ids(t) for t in tokenized_test]
    ids_test = np.array([np.pad(i, (0, MAX_LEN_TEST - len(i)), mode='constant') for i in ids_test])

    print("\nThe indices of the first training sentence:")
    print(ids_train[0])

    amasks_train, amasks_test = [], []

    for seq in ids_train:
        seq_mask = [float(i > 0) for i in seq]
        amasks_train.append(seq_mask)

    for seq in ids_test:
        seq_mask = [float(i > 0) for i in seq]
        amasks_test.append(seq_mask)

    print("\nFirst training embedding with attention masks")
    print(amasks_train[0])

    (train_inputs, validation_inputs, train_labels, validation_labels) = train_test_split(
        ids_train, labels_train, random_state=42, test_size=0.1)
    (train_masks, validation_masks, _, _) = train_test_split(
        amasks_train, ids_train, random_state=42, test_size=0.1)

    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels.astype('int64'))
    train_masks = torch.tensor(train_masks)
    validation_inputs = torch.tensor(validation_inputs)
    validation_labels = torch.tensor(validation_labels.astype('int64'))
    validation_masks = torch.tensor(validation_masks)
    test_inputs = torch.tensor(ids_test)
    test_labels = torch.tensor(labels_test.astype('int64'))
    test_masks = torch.tensor(amasks_test)

    num_labels = 615

    # Ensure all label values are within the valid range
    print("Min train label:", train_labels.min())
    print("Max train label:", train_labels.max())

    # Find and print out-of-range labels
    out_of_range_labels = train_labels[(train_labels < 0) | (train_labels >= num_labels)]
    print("Out-of-range train labels:", out_of_range_labels)
    
    assert train_labels.min() >= 0 and train_labels.max() < num_labels, "Train labels are out of range"
    assert validation_labels.min() >= 0 and validation_labels.max() < num_labels, "Validation labels are out of range"
    assert test_labels.min() >= 0 and test_labels.max() < num_labels, "Test labels are out of range"

    # Print the unique labels to inspect
    print("Unique train labels:", np.unique(train_labels))
    print("Unique validation labels:", np.unique(validation_labels))
    print("Unique test labels:", np.unique(test_labels))

    BATCH_SIZE = 32

    print('\nDatasets:')
    print('Train: ', end="")
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
    print(f'{len(train_data)} reviews')

    print('Validation: ', end="")
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)
    print(f'{len(validation_data)} reviews')

    print('Test: ', end="")
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)
    print(f'{len(test_data)} reviews')

    model = BertForSequenceClassification.from_pretrained(BERTMODEL, num_labels=num_labels)
    model.cuda()
    print(f'\nPretrained BERT model "{BERTMODEL}" loaded')

    EPOCHS = 4
    WEIGHT_DECAY = 0.01
    LR = 3e-4
    WARMUP_STEPS = int(0.2 * len(train_dataloader))

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': WEIGHT_DECAY},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=-1)

    model.save_pretrained(ckpt_path)

    train_lossv = []
    for epoch in range(1, EPOCHS + 1):
        print()
        train(epoch, train_lossv)
        print('\nValidation set:')
        evaluate(validation_dataloader)
        print("saving model")
        model.save_pretrained(ckpt_path)

    plt.figure(figsize=(15, 8))
    plt.title("Training loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(train_lossv, label='original')
    plt.plot(np.convolve(train_lossv, np.ones(101), 'same') / 101, label='averaged')
    plt.legend(loc='best')
    plt.savefig("training-loss.png")

    print('\nTest set:')
    evaluate(test_dataloader)
