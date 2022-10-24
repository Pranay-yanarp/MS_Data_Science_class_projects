
#============================================

# 5. apply to remove special characters

def remove_characters(text):
    return re.sub('[^a-zA-Z]', ' ', text)

df['text'] = df['text'].apply(remove_characters)

#============================================

# 6. apply to remove stopwords

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

def remove_stopwords(text):
    return ' '.join([word for word in nltk.word_tokenize(text) if word not in stop_words])

df['text'] = df['text'].apply(remove_stopwords)

#============================================

# 7. apply stemming

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def stemming_words(text):
    return ' '.join(stemmer.stem(word) for word in text.split())

df['text'] = df['text'].apply(stemming_words)

#============================================

# 8. apply lemmatization

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):
    return ' '.join(lemmatizer.lemmatize(word) for word in text.split())

df['text'] = df['text'].apply(lemmatize_words)

df.to_csv(path1+'/'+'new_df.csv')
#============================================







import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import sklearn
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from transformers import DebertaTokenizer, DebertaForSequenceClassification, AdamW
import os

path1 = os.path.split(os.getcwd())[0] + '/project_datasets'

np.random.seed(0)

df = pd.read_csv(path1+'/'+'new_df.csv') # reading csv file


print(df.isnull().sum()) # check if there is null values
df.dropna(inplace=True) # Dropping Null values

df.drop(['Unnamed: 0'],axis=1,inplace=True) # Dropping unnamed column

train_df, sub_df = train_test_split(df, stratify=df.target.values,
                                                  random_state=1,
                                                  test_size=0.2, shuffle=True) # making train and sub df split

validation_df, test_df = train_test_split(sub_df, stratify=sub_df.target.values,
                                                  random_state=1,
                                                  test_size=0.25, shuffle=True) # making validation and test df splits

# resetting index to distribute the data
train_df.reset_index(drop=True, inplace=True)
validation_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

# We check the number of examples after split
print("Train data: {} \n".format(train_df.shape))
print("Validation data: {} \n".format(validation_df.shape))
print("Test data: {} \n".format(test_df.shape))


#DEBERTA
checkpoint = "microsoft/deberta-base"
tokenizer = DebertaTokenizer.from_pretrained(checkpoint)

# Initializing the hyperparameters

NUM_LABELS = 2
BATCH_SIZE = 4
MAX_LEN = 256
EPOCHS = 3
LEARNING_RATE = 1e-5

# Generating id's and attenton mask from train and validation dataframe
train = tokenizer(list(train_df.text.values), truncation=True, padding=True, max_length=MAX_LEN)
train_input_ids = train['input_ids']
train_masks = train['attention_mask']

validation = tokenizer(list(validation_df.text.values), truncation=True, padding=True, max_length=MAX_LEN)
validation_input_ids = validation['input_ids']
validation_masks = validation['attention_mask']

# Train data TO TENSOR
train_inputs = torch.tensor(train_input_ids)
train_masks = torch.tensor(train_masks)
train_labels = torch.tensor(train_df.target.values)

# Validation data to TENSOR
validation_labels = torch.tensor(validation_df.target.values)
validation_inputs = torch.tensor(validation_input_ids)
validation_masks = torch.tensor(validation_masks)


# DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

# DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # If gpu is available will be trained on GPU
print("Using device {}.\n".format(device))

# Model Building
model = DebertaForSequenceClassification.from_pretrained(checkpoint, num_labels=NUM_LABELS, output_hidden_states=False, output_attentions=False)
model = model.to(device) # copying all tensor variables to GPU as specified by the device

optimizer = AdamW(model.parameters(),lr=LEARNING_RATE)  # Optimization of model

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * EPOCHS

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value
                                            num_training_steps = total_steps)

def count_parameters(model):
    ''' Count the total number of trainable parameters '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(model)
print("The model has total {} trainable parameters".format(num_params))


def eval_metric(predictions, labels):
  ''' Calculate average accuracy of the data samples '''
  max_predictions = predictions.argmax(axis=1, keepdim=True)
  avg_acc = round(accuracy_score(y_true=labels.to('cpu').tolist(), y_pred=max_predictions.detach().cpu().numpy()), 2)*100
  return avg_acc

# Training Model
def train_fn(model, train_loader, optimizer, device, scheduler, criterion=None):
  ''' Define the training function '''
  model.train() #set the model on train mode
  total_loss, total_acc = 0, 0

  for batch in train_loader:
    input_ids = batch[0].to(device)
    input_mask = batch[1].to(device)
    labels = batch[2].to(device)
    optimizer.zero_grad() # Removing gradients from last batch
    outputs = model(input_ids, attention_mask=input_mask, labels=labels) # Retrieve Predictions
    loss = outputs.loss
    total_loss += loss.item() # Average of losses
    loss.backward() # Compute the gradients
    optimizer.step() # Updating parameters
    scheduler.step()
    logits = outputs.logits # Compute logits
    total_acc += eval_metric(logits, labels)

  loss_per_epoch = total_loss/len(train_loader) # Compute loss per epoch
  acc_per_epoch = total_acc/len(train_loader)
  return loss_per_epoch, acc_per_epoch

# Evaluation function

def eval_fn(model, data_loader, device, criterion=None):
  ''' Define the evaluation function '''
  model.eval() # set the model on eval mode
  total_loss, total_acc = 0, 0

  with torch.no_grad():
    for batch in data_loader:
      input_ids = batch[0].to(device)
      input_mask = batch[1].to(device)
      labels = batch[2].to(device)
      outputs = model(input_ids, attention_mask=input_mask, labels=labels) # get predictions
      loss = outputs.loss
      total_loss += loss.item() # Average of losses
      logits = outputs.logits
      total_acc += eval_metric(logits, labels)

  loss_per_epoch = total_loss/len(data_loader)
  acc_per_epoch = total_acc/len(data_loader)
  return loss_per_epoch, acc_per_epoch

# Initializing Train and validation losses for plotting

train_losses = []
validation_losses = []

train_accuracies = []
validation_accuracies = []

best_val_loss = float('inf')

for epoch in range(EPOCHS):
    train_loss_per_epoch, train_acc_per_epoch = train_fn(model, train_dataloader, optimizer, device, scheduler)
    val_loss_per_epoch, val_acc_per_epoch = eval_fn(model, validation_dataloader, device)

    train_losses.append(train_loss_per_epoch)
    validation_losses.append(val_loss_per_epoch)
    train_accuracies.append(train_acc_per_epoch)
    validation_accuracies.append(val_acc_per_epoch)

    if val_loss_per_epoch < best_val_loss:
        best_val_loss = val_loss_per_epoch
        torch.save(model.state_dict(), 'model_2.pt')

    print("Epoch: {}, Train Loss: {:.4f}, Train Accuracy: {:.2f}%".format(epoch, train_loss_per_epoch,
                                                                          train_acc_per_epoch))
    print("Epoch: {}, Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%\n".format(epoch, val_loss_per_epoch,
                                                                                      val_acc_per_epoch))

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

# Loading best trained model to interpret results

print("Loading the best trained model")
model.load_state_dict(torch.load('model_2.pt'))

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