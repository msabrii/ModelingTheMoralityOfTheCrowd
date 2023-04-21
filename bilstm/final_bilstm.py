# -*- coding: utf-8 -*-

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import argparse
import optuna
import gc
import warnings

from torch import nn 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')


# The models to use for sentence embeddings
models = {
   'mpnet': "all-mpnet-base-v2",
   'roberta': "all-distilroberta-v1",
   'minilm': "all-MiniLM-L12-v2"
}

# The two available datasets
datasets = {
   "2": "subset2",
   "3": "subset3",
   "combined": "combined"
}

# Set the device for PyTorch based on whether CUDA is available or not
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.BCELoss()

def get_embeddings(embedding, post):
  sentences = sent_tokenize(post)
  final = []

  # For each sentence in a post, tokenize and then stack all tensors together
  for sent in sentences:
    text_emb = embedding.encode(sent,show_progress_bar=False, convert_to_tensor=True)
    final.append(text_emb)
  return torch.stack(final).cpu()

# EarlyStopper class
class EarlyStopper:
  def __init__(self, patience=1, min_delta=0):
    """
    Initialize the EarlyStopper object with the specified parameters.

    Parameters:
      patience (int): Number of epochs to wait before stopping if validation loss does not improve.
      min_delta (float): Minimum change in validation loss to be considered an improvement.
    """
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.min_validation_loss = np.inf

  def early_stop(self, validation_loss):
    """
    Check if early stopping criteria is met based on the current validation loss.

    Parameters:
      validation_loss (float): Current validation loss.

    Returns:
      bool: True if early stopping criteria is met, False otherwise.
    """
    # Check if the validation loss has improved
    if validation_loss < self.min_validation_loss:
        self.min_validation_loss = validation_loss
        self.counter = 0
    # Check if the validation loss has not improved by the minimum delta
    elif validation_loss > (self.min_validation_loss + self.min_delta):
        self.counter += 1
        # Check if the patience limit has been reached
        if self.counter >= self.patience:
            return True
    return False

# Dataset class
class MyDataset(Dataset):
  def __init__(self, df):
    x = df['sent_tokens'].values
    y = df['label'].values

    self.x_train = x
    self.y_train = torch.tensor(y,dtype=torch.float32).to(device)

  def __len__(self):
    return len(self.y_train)
    
  def __getitem__(self,idx):
    return self.x_train[idx], self.y_train[idx]

# Custom collate function
def mycollator(batch):
  # We pad the sequences in the LSTM's forward pass
  # Default collator does not allow tensors to be of different sizes
  # But in our case it is okay to have tensors of different sizes as they will be padded to equal length in the forward pass
  new_x = []
  new_y = []
  for x,y in batch:
    new_x.append(x.to(device))
    new_y.append(y)
  return [new_x, torch.tensor(new_y,dtype=torch.float32).to(device)]

# LSTM class
class LSTM(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, max_pool=False, dropout_val=0.3, concatenate_hidden_states=False):
    super().__init__()
    self.lstm = nn.LSTM(embedding_dim, 
                      hidden_dim, 
                      num_layers=1, 
                      bidirectional=True,
                      batch_first=True
                      )
    self.dropout = torch.nn.Dropout(dropout_val if dropout_val != None else 0)
    self.fc = nn.Linear(hidden_dim * 2, 1)
    self.act = nn.Sigmoid()
    self.max_pool = max_pool # Use max pooling, if set to False use mean pooling
    self.concatenate_hidden_states = concatenate_hidden_states # Concatenate final hidden states, if set to False use pooled output

  def forward(self, text):
    text_emb = torch.nn.utils.rnn.pad_sequence(text, batch_first=True) # Pad batched input to the same sequence length
    output, (ht, ct) = self.lstm(text_emb)

    if(self.concatenate_hidden_states):
      h_1, h_2 = ht[0], ht[1]
      pooled_output = torch.cat((h_1, h_2), 1)
    else:
      if(self.max_pool):
        pooled_output, _ = torch.max(output, 1) # Max Pool
      else:
        pooled_output = torch.mean(output, 1) # Average Pool

    text_fea = self.dropout(pooled_output)
    text_fea = self.fc(text_fea)
    text_out = torch.sigmoid(text_fea)

    return text_out.squeeze(1)

def train_model(model, train_loader, val_loader, epochs=30, lr=0.001, early_stop=False):
  args = getArgs()

  # Set the optimizer and learning rate scheduler
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  early_stopper = EarlyStopper(patience=1, min_delta=0.01)

  model.zero_grad()
  # Fine-tune the model
  for epoch in range(epochs):
      model.train()
      sum_loss = 0.0
      total = 0
      predictions = []
      true_labels = []
      for input_ids, labels in train_loader:
        output = model(input_ids)
        pred = (output > 0.5).long()
        loss = criterion(output, labels.to(device))

        # Zero the gradients, backpropagate, and update the parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.zero_grad()
        gc.collect()      

        sum_loss += loss.item()*labels.shape[0]
        total += labels.shape[0]
        predictions.extend(pred.cpu().numpy())
        true_labels.extend(labels.cpu().numpy().astype(int))
      accuracy = accuracy_score(true_labels, predictions)
      mcc_score = mcc(true_labels, predictions)

      if args.validate_after_every_epoch:
        val_loss, val_acc, mcc_score = validation_metrics(model, val_loader)
        print(f'Training Epoch Number {epoch} Train Loss={sum_loss/total:.4f}, Train Accuracy={accuracy}, Validation Loss: {val_loss:.4f}, Validation Accuracy {val_acc}, Validation MCC={mcc_score:.4f}')

      elif epoch % 2 == 0:
        val_loss, val_acc, mcc_score = validation_metrics(model, val_loader)
        print(f'Training Epoch Number {epoch} Train Loss={sum_loss/total:.4f}, Train Accuracy={accuracy}, Validation Loss: {val_loss:.4f}, Validation Accuracy {val_acc}, Validation MCC={mcc_score:.4f}')
      
      # Early Stop to prevent overfitting
      if epoch % 2 and early_stop and early_stopper.early_stop(val_loss):    # Early Stop to prevent overfitting
          print('EARLY STOP')         
          break
      
  return accuracy

def validation_metrics (model, valid_dl):
    model.eval()
    total = 0
    sum_loss = 0.0
    predictions = []
    true_labels = []
    for input_ids, labels in valid_dl:
        # Compute model outputs and loss for the current batch
        output = model(input_ids)
        loss = criterion(output, labels)
        pred = (output > 0.5).long()
        total += labels.shape[0]
        sum_loss += loss.item()*labels.shape[0]
        predictions.extend(pred.cpu())
        true_labels.extend(labels.cpu())

    accuracy = accuracy_score(true_labels, predictions)
    mcc_score = mcc(true_labels, predictions)
    return sum_loss/total, accuracy, mcc_score

def test_model(model, test_loader):
  model.eval()
  total = 0
  sum_loss = 0.0
  predictions = []
  true_labels = []
  for input_ids, labels in test_loader:
      # Compute model outputs and loss for the current batch
      output = model(input_ids)
      loss = criterion(output, labels)
      pred = (output > 0.5).long()
      predictions.extend(pred.cpu())
      true_labels.extend(labels.cpu())
      total += labels.shape[0]
      sum_loss += loss.item()*labels.shape[0]

  # Compute average loss, accuracy and mcc across all batches
  tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
  accuracy = accuracy_score(true_labels, predictions)
  mcc_score = mcc(true_labels, predictions)
  f1Score = f1_score(true_labels, predictions)
  avg_loss = sum_loss/total

  print(f'Testing Model : Loss={avg_loss:.4f}, Accuracy={accuracy}, MCC={mcc_score:.4f}, F1 Score={f1Score:.4f}')
  print('True Positives:', tp, 'False Positives:', fp, 'True Negatives:', tn, 'False Negatives:', fn)

# Callback function for optuna to keep track of the best accuracy so far
def callback(study, trial):
  if study.best_trial.number == trial.number:
      current_acc = trial.user_attrs["current_acc"]
      study.set_user_attr(key="best_acc", value=current_acc)

# Function to free up memory
def garbage(study, trial):
  gc.collect()

# Objective function used by optuna to find the best set of hyperparameters
def objective(trial, study, args, train_loader, val_loader):
  # batch_size = trial.suggest_int("batch_size", 4, 32, 4)
  # epoc = trial.suggest_int("epochs", 5, 50, 5)

  h_layers = trial.suggest_int("hidden_layers", 10, 300, 10)
  lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
  pooling = trial.suggest_int("pooling", 0, 1)
  dropout_val = trial.suggest_float("dropout_val", 0, 0.5, step=0.1)
  concatenate_final_states = trial.suggest_int("concatenate_hidden_states", 0, 1)
  model = LSTM(args.hidden_dim, h_layers, max_pool=pooling, dropout_val=dropout_val, concatenate_hidden_states=concatenate_final_states).to(device)
  train_acc = train_model(model, epochs=30, early_stop=args.early_stop, lr=lr, train_loader=train_loader, val_loader=val_loader)
  _, val_acc, _ = validation_metrics(model, val_loader)

  trial.set_user_attr(key="current_acc", value=val_acc)
  trial.set_user_attr(key="train_acc", value=train_acc)
  best_acc = study.user_attrs["best_acc"]

  if(val_acc > best_acc):
    if not os.path.exists(os.getcwd() + '/bilstm_models/'):
      os.makedirs(os.getcwd() + '/bilstm_models/')
    with open(os.getcwd() + "/bilstm_models/bilstm-{}-subset{}-{}.pt".format(args.model, args.subset, trial.number), "wb") as fout:
        torch.save(model, fout)
      
  return val_acc

def set_seed(seed):
  # Set the seed values for reproducibility
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)

def getArgs():
  # Add arguments to the parser object
  parser = argparse.ArgumentParser()
  parser.add_argument('--subset', type=str, required=True, help="The dataset to use", choices={"2", "3", "combined"}, default="2")
  parser.add_argument('--model', type=str, required=True, help="The model to use for sentence embeddings", choices={"roberta", "mpnet", "minilm"}, default='roberta')
  parser.add_argument('--num_epochs', type=int, help="The number of epochs to train for", default=5)
  parser.add_argument('--num_hidden_layers', type=int, help="The number of hidden layers for the biLSTM", default=100)
  parser.add_argument('--batch_size', type=int, help="The batch size to use", default=8)
  parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
  parser.add_argument("--dropout", type=float, default=None, help="Dropout rate")
  parser.add_argument("--concatenate_hidden_states", help="If this flag is set LSTM will concatenate the hidden states, if this flag is not set, use pooled output", action='store_true')
  parser.add_argument('--validate_after_every_epoch', help="Runs validation after every epoch", action='store_true')
  parser.add_argument('--early_stop', help="Use early stopping in model training", action='store_true')    
  parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
  parser.add_argument('--optuna', help="run optuna to find best hyperparams", action='store_true')
  parser.add_argument("--num_optuna_trials", type=int, default=20, help="Number of trials to run for Optuna")
  parser.add_argument("--out_dir", type=str, default="bilstm_output", help="The output directory where the model will be stored")

  # Parse the arguments
  args = parser.parse_args()

  # Set a seed for reproducibility
  set_seed(args.seed)
  return args
  
def main():
  args = getArgs()
  
  dataset = datasets[args.subset]
  print(args.subset, args.model, 'ARGS')

  # MiniLM has an input dimension of 384
  args.hidden_dim = 768 if args.model != 'minilm' else 384

  # Get the sentence transformer to use
  sentenceTransformer = models[args.model]
  embedding = SentenceTransformer(sentenceTransformer)

  # Get the dataset
  dataset = datasets[args.subset]

  # Read the train, test and val splits
  df_train = pd.read_csv('./' + dataset + '_train.csv')
  df_test = pd.read_csv('./' + dataset + '_test.csv')
  df_val = pd.read_csv('./' + dataset + '_val.csv')

  # Generate text embeddings
  df_train['sent_tokens'] = df_train.apply(lambda x: get_embeddings(embedding, x['body']), axis=1)
  df_test['sent_tokens'] = df_test.apply(lambda x: get_embeddings(embedding, x['body']), axis=1)
  df_val['sent_tokens'] = df_val.apply(lambda x: get_embeddings(embedding, x['body']), axis=1)

  batch_size = args.batch_size

  train_dataset= MyDataset(df_train)
  test_dataset= MyDataset(df_test)
  val_dataset = MyDataset(df_val)

  train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=mycollator)
  val_loader = DataLoader(val_dataset, batch_size, shuffle=True, collate_fn=mycollator)
  test_loader = DataLoader(test_dataset, batch_size, shuffle=True, collate_fn=mycollator)

  # If optuna flag is enabled perform hyperparameter optimization
  if(args.optuna):
    # Create optuna study
    STUDY_NAME = "bilstm-" + args.model+"-subset"+str(args.subset)
    study = optuna.create_study(direction='maximize', study_name=STUDY_NAME, storage='sqlite:///' + STUDY_NAME + '.db', load_if_exists=True)
    
    # Initialzie best_acc var to 0
    if(study.user_attrs.get('best_acc') is None):
      study.set_user_attr("best_acc", 0)
    
    # Create a wrapper function to be able to pass more parameters to the objective function
    func = lambda trial: objective(trial, study, args, train_loader, val_loader)

    # If trials have already been conducted, print the current best and total number of trials
    if(len(study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))) > 1):
      print(study.best_trial)
      print('Total number of trials', len(study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))))

    # Enque first trail
    study.enqueue_trial({"hidden_layers": 200, "lr": 2e-5, "pooling": 0, "dropout_val": 0.2, "concatenate_hidden_states": 1}, skip_if_exists=True)

    # Start Optuna Optimization
    study.optimize(func, n_trials=args.num_optuna_trials, show_progress_bar=True, callbacks=[callback, garbage])

    # Print results of optimization
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
      print("    {}: {}".format(key, value))
    print('Train Accuracy: ', trial.user_attrs['train_acc'])

    # Load best model and perform testing
    model = torch.load(os.getcwd() + '/bilstm_models/' + STUDY_NAME + '-' + str(study.best_trial.number) + '.pt')
    print('TESTING MODE')
    test_model(model, test_loader)
  else:
    # Initialize model, then train and test
    model = LSTM(args.hidden_dim, args.num_hidden_layers, concatenate_hidden_states=args.concatenate_hidden_states, dropout_val=args.dropout).to(device)
    train_model(model, train_loader, val_loader, epochs=args.num_epochs, early_stop=args.early_stop, lr=args.learning_rate)
    test_model(model, test_loader)
    if not os.path.exists(os.getcwd() + "/" + args.out_dir):
      os.makedirs(os.getcwd() + "/" + args.out_dir)
    with open(os.getcwd() + "/" + args.out_dir + "/bilstm-{}-subset{}.pt".format(args.model, args.subset), "wb") as fout:
      torch.save(model, fout)
    

if __name__ == "__main__":
   with warnings.catch_warnings(record=True):
    main()
