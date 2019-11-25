# https://towardsdatascience.com/bert-for-dummies-step-by-step-tutorial-fb90890ffe03

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import io

from nlu_seq2seq_load_data import load_atis

def run():
    # check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    print(f'Device: {device}, GPU Count: {n_gpu}, Name: {device_name}')

    # load data
    t2i_train, s2i_train, in2i_train, i2t_train, i2s_train, i2in_train, \
        input_tensor_train, target_tensor_train, \
        query_data_train, intent_data_train, intent_data_label_train, slot_data_train \
        = load_atis('ms_cntk_atis.train.pkl', verbose=False)
    t2i_test, s2i_test, in2i_test, i2t_test, i2s_test, i2in_test, \
        input_tensor_test, target_tensor_test, \
        query_data_test, intent_data_test, intent_data_label_test, slot_data_test \
        = load_atis('ms_cntk_atis.test.pkl', verbose=False)

    # label adjustment
    labels = intent_data_label_train
    nb_labels = len(np.unique(labels))
    labels[labels == 14] = -1
    labels[labels != -1] = 0
    labels[labels == -1] = 1

    # add special token for BERT
    sentences = ["[CLS] " + query + " [SEP]" for query in query_data_train]

    # tokenize with BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    # print("Tokenize text first sentence: ", tokenized_texts[0])

    MAX_LEN = 128
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')

    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    train_inputs, validation_inputs, train_labels, validation_labels = \
        train_test_split(input_ids, labels, random_state=2018, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2018, test_size=0.1)

    # convert data into torch tensors
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    batch_size = 32

    # Create an iterator of data with torch DataLoader
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = RandomSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=nb_labels)
    model.cuda()

    # fine-tune params
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=2e-5, warmup=.1)

    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    train_loss_set = []
    epochs = 4

    for _ in trange(epochs, desc='Epoch'):
        # training
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer.zero_grad()
            loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            train_loss_set.append(loss.item())
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
        print('Train loss: {}'.format(tr_loss / nb_tr_steps))

        # validation
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
        print('Validation Accuracy: {}'.format(eval_accuracy / nb_eval_steps))

    # Model prediction
    labels = intent_data_label_test
    nb_labels = len(np.unique(labels))
    labels[labels == 14] = -1
    labels[labels != -1] = 0
    labels[labels == -1] = 1

    sentences = ["[CLS] " + query + " [SEP]" for query in query_data_test]
    labels = intent_data_label_test
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    MAX_LEN = 128
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(labels)
    batch_size = 32
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = RandomSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    model.eval()
    predictions, true_labels = [], []
    for batch in prediction_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.append(logits)
        true_labels.append(label_ids)

    from sklearn.metrics import matthews_corrcoef
    matthews_set = []
    for i in range(len(true_labels)):
        matthews = matthews_corrcoef(true_labels[i], np.argmax(predictions[i], axis=1).flatten())
        matthews_set.append(matthews)
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    print('Classification accuracy using BERT: {0:0.2%}'.format(matthews_corrcoef(flat_true_labels, flat_predictions)))


if __name__ == '__main__':
    run()
