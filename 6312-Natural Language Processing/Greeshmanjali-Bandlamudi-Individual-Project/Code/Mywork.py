from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import os

import nltk # NLP
import re

import os

path1 = os.path.split(os.getcwd())[0] + '/project_datasets'

np.random.seed(0)

Fake_df = pd.read_csv(path1+'/'+'Fake.csv')
True_df = pd.read_csv(path1+'/'+'True.csv')


Fake_df['target'] = 0
True_df['target'] = 1

df = pd.concat([True_df, Fake_df], ignore_index=True, sort=False)
import random
df = df.sample(len(df),ignore_index=True)

df['text'] = df['title'] + ' ' + df['text']

df.drop(['title', 'subject', 'date',], axis=1, inplace=True)

# tokenize and cleaning

#  1. apply to remove url
def remove_url(text):
    return re.sub(r'http\S+', '', text)

df['text'] = df['text'].apply(remove_url)

#============================================

# 2. apply lower
def to_lower(text):
    return text.lower()

df['text'] = df['text'].apply(to_lower)

#============================================

# 3. apply contractions

# import sys
# import subprocess

# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'contractions'])

# subprocess.call("pip install --upgrade " + 'pip', shell=True)

import contractions

def remove_contractions(text):
    return ' '.join([contractions.fix(word) for word in text.split()])

df['text'] = df['text'].apply(remove_contractions)


#============================================

# 4. apply remove punctuations

def remove_punctuations(text):
    return re.sub('\[[^]]*\]', '', text)

df['text'] = df['text'].apply(remove_punctuations)







import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
import sklearn
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
import os
# Loading the dataset
path1 = os.path.split(os.getcwd())[0] + '/project_datasets'
np.random.seed(0)
df = pd.read_csv(path1+'/'+'new_df.csv')
# checking for null values
print(df.isnull().sum())
# dropped the null values if present
df.dropna(inplace=True)
df.drop(['Unnamed: 0'],axis=1,inplace=True)
# split the dataset to train, validation, test
train_df, sub_df = train_test_split(df, stratify=df.target.values, random_state=1, test_size=0.2, shuffle=True)
validation_df, test_df = train_test_split(sub_df, stratify=sub_df.target.values, random_state=1, test_size=0.25, shuffle=True)
train_df.reset_index(drop=True, inplace=True)
validation_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
# We check the number of examples after split
print("Train data: {} \n".format(train_df.shape))
print("Validation data: {} \n".format(validation_df.shape))
print("Test data: {} \n".format(test_df.shape))
# RoBERTa
checkpoint = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(checkpoint, do_lower_case=True)
NUM_LABELS = 2
BATCH_SIZE = 32
MAX_LEN = 256
N_EPOCHS = 3
LEARNING_RATE = 1e-5
train = tokenizer(list(train_df.text.values), truncation=True, padding=True, max_length=MAX_LEN)
train_input_ids = train['input_ids']
train_masks = train['attention_mask']

validation = tokenizer(list(validation_df.text.values), truncation=True, padding=True, max_length=MAX_LEN)
validation_input_ids = validation['input_ids']
validation_masks = validation['attention_mask']
#TO TENSOR
train_inputs = torch.tensor(train_input_ids)
validation_inputs = torch.tensor(validation_input_ids)
train_labels = torch.tensor(train_df.target.values)

validation_labels = torch.tensor(validation_df.target.values)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)
# DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
# DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device {}.\n".format(device))
model = RobertaForSequenceClassification.from_pretrained(checkpoint, num_labels=NUM_LABELS, output_hidden_states=False, output_attentions=False)
model = model.to(device)
optimizer = AdamW(model.parameters(),lr=LEARNING_RATE)
# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * N_EPOCHS
# Create the learning rate scheduler.
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
num_params = count_parameters(model)
print("The model has total {} trainable parameters".format(num_params))
def eval_metric(predictions, labels):
  max_predictions = predictions.argmax(axis=1, keepdim=True)
  avg_acc = round(accuracy_score(y_true=labels.to('cpu').tolist(), y_pred=max_predictions.detach().cpu().numpy()), 2)*100
  return avg_acc
def train_fn(model, train_loader, optimizer, device, lr_scheduler):
# defining the training section
  model.train()
  total_loss, total_acc = 0, 0
  for batch in train_loader:
    batch = tuple(item.to(device) for item in batch)
    input_ids = batch[0].to(device)
    input_mask = batch[1].to(device)
    labels = batch[2].to(device)
# input_ids, input_mask, labels = batch
    optimizer.zero_grad() # clear gradients from last batch
    outputs = model(input_ids, attention_mask=input_mask, labels=labels) # get predictions
    loss = outputs.loss
    total_loss += loss.item() #aggregate the losses
    loss.backward() # compute the gradients
    optimizer.step() # update the parameters
    lr_scheduler.step()
    logits = outputs.logits
    total_acc += eval_metric(logits, labels)
  loss_per_epoch = total_loss/len(train_loader)
  acc_per_epoch = total_acc/len(train_loader)
  return loss_per_epoch, acc_per_epoch
def eval_fn(model, data_loader, device):
  model.eval() # set the model on eval mode
  total_loss, total_acc = 0, 0
  with torch.no_grad(): # do not need to update the parameters
    for batch in data_loader:
      batch = tuple(item.to(device) for item in batch)
      input_ids = batch[0].to(device)
      input_mask = batch[1].to(device)
      labels = batch[2].to(device)
      # input_ids, input_mask, labels = batch
      outputs = model(input_ids, attention_mask=input_mask, labels=labels) # get predictions
      loss = outputs.loss
      total_loss += loss.item() #aggregate the losses
      logits = outputs.logits
      total_acc += eval_metric(logits, labels)
  loss_per_epoch = total_loss/len(data_loader)
  acc_per_epoch = total_acc/len(data_loader)
  return loss_per_epoch, acc_per_epoch
 # Store train and validation losses for plotting
train_losses = []
validation_losses = []
train_accuracies = []
validation_accuracies = []
best_val_loss = float('inf')
for epoch in range(N_EPOCHS):
    train_loss_per_epoch, train_acc_per_epoch = train_fn(model, train_dataloader, optimizer, device, lr_scheduler)
    val_loss_per_epoch, val_acc_per_epoch = eval_fn(model, validation_dataloader, device)
    train_losses.append(train_loss_per_epoch)
    validation_losses.append(val_loss_per_epoch)
    train_accuracies.append(train_acc_per_epoch)
    validation_accuracies.append(val_acc_per_epoch)
    if val_loss_per_epoch < best_val_loss:
        best_val_loss = val_loss_per_epoch
        torch.save(model.state_dict(), 'model.pt')
    print("Epoch: {}, Train Loss: {:.4f}, Train Accuracy: {:.2f}%".format(epoch, train_loss_per_epoch,
                                                                          train_acc_per_epoch))
    print("Epoch: {}, Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%\n".format(epoch, val_loss_per_epoch,
                                                                                      val_acc_per_epoch))
# Plotting the graphs
import matplotlib.pyplot as plt
plt.plot(np.arange(1,4),validation_losses)
plt.title('validation loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('validation loss')
plt.show()
plt.plot(np.arange(1,4),train_losses)
plt.title('train loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('train loss')
plt.show()
plt.plot(np.arange(1,4),train_accuracies)
plt.title('train accuracies vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('train_accuracy')
plt.show()
plt.plot(np.arange(1,4),validation_accuracies)
plt.title('validation accuracies vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('validation_accuracy')
plt.show()
# Test
# load the best training model
print("Loading the best trained model")
model.load_state_dict(torch.load('model.pt'))
test = tokenizer(list(test_df[:1000].text.values), truncation=True, padding=True, max_length=30)
test_input_ids = test['input_ids']
test_masks = test['attention_mask']
test_masks = torch.tensor(test_masks)
test_input_ids = torch.tensor(test_input_ids)
with torch.no_grad():
    test_input_ids = test_input_ids.to(device)
    test_masks = test_masks.to(device)
    outputs = model(test_input_ids, test_masks)
    logits = outputs.logits  # output[0] #[batch_size, num_classes]
    batch_logits = logits.detach().cpu().numpy()  # shape: [batch_size, num_classes]
    preds = np.argmax(batch_logits, axis=1)
print(classification_report(test_df[:1000].target.values, preds))
print("ROC AUC Score: {}".format(roc_auc_score(y_true=test_df[:1000].target.values, y_score=preds)))
print('f1-score of the model:',sklearn.metrics.f1_score(test_df[:1000].target.values, preds))