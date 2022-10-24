import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

import os

path1 = os.path.split(os.getcwd())[0] + '/project_datasets'

np.random.seed(0)

df = pd.read_csv(path1+'/'+'new_df.csv')


print(df.isnull().sum())
df.dropna(inplace=True)

df.drop(['Unnamed: 0'],axis=1,inplace=True)

train_df, sub_df = train_test_split(df, stratify=df.target.values,
                                                  random_state=1,
                                                  test_size=0.2, shuffle=True)

validation_df, test_df = train_test_split(sub_df, stratify=sub_df.target.values,
                                                  random_state=1,
                                                  test_size=0.25, shuffle=True)


train_df.reset_index(drop=True, inplace=True)
validation_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

#=========================================================================

# print the number of observations in train, validations and test dataframes
print("Train data: {} \n".format(train_df.shape))
print("Validation data: {} \n".format(validation_df.shape))
print("Test data: {} \n".format(test_df.shape))

#=========================================================================

# distilbert

checkpoint = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(checkpoint, do_lower_case=True)

NUM_LABELS = 2
BATCH_SIZE = 32
MAX_LEN = 256
EPOCHS = 3
LEARNING_RATE = 1e-5

train = tokenizer(list(train_df.text.values), truncation=True, padding=True, max_length=MAX_LEN)
train_input_ids = train['input_ids']
train_masks = train['attention_mask']

validation = tokenizer(list(validation_df.text.values), truncation=True, padding=True, max_length=MAX_LEN)
validation_input_ids = validation['input_ids']
validation_masks = validation['attention_mask']

#=========================================================================================================

# transform the train and validation data inputs to tensor format
train_inputs = torch.tensor(train_input_ids)
validation_inputs = torch.tensor(validation_input_ids)

train_labels = torch.tensor(train_df.target.values)
validation_labels = torch.tensor(validation_df.target.values)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

#=========================================================================================================

# Use dataloader to send the inputs in BATCH_SIZE chunks to the model
# Train dataloader
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

# validation dataloader
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)

#=========================================================================================================

# select device to run on GPU, if not run on CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device {}.\n".format(device))

#=========================================================================================================
# modelling
model = DistilBertForSequenceClassification.from_pretrained(checkpoint, num_labels=NUM_LABELS, output_hidden_states=False, output_attentions=False)
model = model.to(device) # move model to gpu

optimizer = AdamW(model.parameters(),lr=LEARNING_RATE) # using AdamW as optimizer

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * EPOCHS

# Using learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

#=========================================================================================================

# count the number of trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(model)
print("The model has total {} trainable parameters".format(num_params))

#=========================================================================================================

# calculate the accuracy score
def eval_metric(predictions, labels):
  max_predictions = predictions.argmax(axis=1, keepdim=True)
  avg_acc = round(accuracy_score(y_true=labels.to('cpu').tolist(), y_pred=max_predictions.detach().cpu().numpy()), 2)*100
  return avg_acc

#=======================================================================================

def train_fn(model, train_loader, optimizer, device, scheduler):
  # train dataset
  model.train()
  total_loss, total_acc = 0, 0

  for batch in train_loader:
    batch = tuple(item.to(device) for item in batch)
    input_ids, input_mask, labels = batch
    optimizer.zero_grad() # create and clear old optimizer
    outputs = model(input_ids, attention_mask=input_mask, labels=labels) # calling the model to get predictions
    loss = outputs.loss
    total_loss += loss.item() #add the losses
    loss.backward() # compute the gradients
    optimizer.step() # update the parameters
    scheduler.step()
    logits = outputs.logits
    total_acc += eval_metric(logits, labels)

  loss_per_epoch = total_loss/len(train_loader)
  acc_per_epoch = total_acc/len(train_loader)
  return loss_per_epoch, acc_per_epoch

#=======================================================================================

def eval_fn(model, data_loader, device):
  # model validation dataset
  model.eval()
  total_loss, total_acc = 0, 0

  with torch.no_grad(): # we don't need tho change the trained parameters
    for batch in data_loader:
      batch = tuple(item.to(device) for item in batch)
      input_ids, input_mask, labels = batch
      outputs = model(input_ids, attention_mask=input_mask, labels=labels) # run the model and get the predictions
      loss = outputs.loss
      total_loss += loss.item() #add the losses
      logits = outputs.logits
      total_acc += eval_metric(logits, labels)

  loss_per_epoch = total_loss/len(data_loader)
  acc_per_epoch = total_acc/len(data_loader)
  return loss_per_epoch, acc_per_epoch

#=======================================================================================

# Store train and validation losses for plotting
# losses
train_losses = []
validation_losses = []

# accuracies
train_accuracies = []
validation_accuracies = []

best_val_loss = float('inf')

for epoch in range(EPOCHS):
    train_loss_per_epoch, train_acc_per_epoch = train_fn(model, train_dataloader, optimizer, device, scheduler)
    val_loss_per_epoch, val_acc_per_epoch = eval_fn(model, validation_dataloader, device)
    # update the losses and accuracies
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

#=======================================================================================

# Plotting the losses and accuracies

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

#=======================================================================================
#=======================================================================================

# Testing
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
    logits = outputs.logits
    batch_logits = logits.detach().cpu().numpy()
    preds = np.argmax(batch_logits, axis=1)

print(classification_report(test_df[:1000].target.values, preds))
print("ROC AUC Score: {}".format(roc_auc_score(y_true=test_df[:1000].target.values, y_score=preds)))
print('f1-score of the model:',sklearn.metrics.f1_score(test_df[:1000].target.values, preds))






