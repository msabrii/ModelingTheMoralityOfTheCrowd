# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import gc
from transformers import LongformerTokenizer, LongformerModel, LongformerConfig, AdamW, get_linear_schedule_with_warmup
import argparse
import optuna
import os
import warnings

# The two available datasets
datasets = {
   "2": "subset2",
   "3": "subset3",
   "combined": "combined"
}

# Set the device for PyTorch based on whether CUDA is available or not
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_data(df, batch_size):
    args = getArgs()
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

    # Extract the 'body' column values and labels from the dataframe
    sentences = df.body.values
    labels = df.label.values

    input_ids = []
    attention_masks = []

    for sent in sentences:
        # Encode each sentence using the tokenizer
        encoded_dict = tokenizer.encode_plus(
                            sent,                      
                            add_special_tokens = True,
                            max_length = args.max_seq_length,          
                            truncation=True,
                            pad_to_max_length = True,
                            return_attention_mask = True,   
                            return_tensors = 'pt', 
                    )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Concatenate the tokenized input and attention masks
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_masks, labels)

    return DataLoader(dataset, batch_size, shuffle=True)

def getConfig():
    args = getArgs()
    config = LongformerConfig.from_pretrained(
      'allenai/longformer-base-4096',
      num_labels=2
    )

    # bert dim is 768.
    hidden_dim = (768 + 384) // 2
    
    # sets the parameters of IB
    config.activation = args.activation
    config.hidden_dim = hidden_dim
    config.ib_dim = args.ib_dim
    config.beta = 1e-05
    config.sample_size = 5
    config.kl_annealing = 'linear'

    if(args.dropout):
        config.hidden_dropout_prob = args.dropout

    return config
 
class ViLongformer(LongformerModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.longformer = LongformerModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ib_dim = config.ib_dim
        self.activation = config.activation
        self.activations = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid()}
        self.kl_annealing = config.kl_annealing
        self.hidden_dim = config.hidden_dim
        intermediate_dim = (self.hidden_dim+config.hidden_size)//2
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, intermediate_dim),
            self.activations[self.activation],
            nn.Linear(intermediate_dim, self.hidden_dim),
            self.activations[self.activation])
        self.beta = config.beta
        self.sample_size = config.sample_size
        self.emb2mu = nn.Linear(self.hidden_dim, self.ib_dim)
        self.emb2std = nn.Linear(self.hidden_dim, self.ib_dim)
        self.mu_p = nn.Parameter(torch.randn(self.ib_dim))
        self.std_p = nn.Parameter(torch.randn(self.ib_dim))
        self.classifier = nn.Linear(self.ib_dim, self.config.num_labels)

        self.init_weights()

    def estimate(self, emb, emb2mu, emb2std):
        """Estimates mu and std from the given input embeddings."""
        mean = emb2mu(emb)
        std = torch.nn.functional.softplus(emb2std(emb))
        return mean, std

    def kl_div(self, mu_q, std_q, mu_p, std_p):
        """Computes the KL divergence between the two given variational distribution.\
           This computes KL(q||p), which is not symmetric. It quantifies how far is\
           The estimated distribution q from the true distribution of p."""
        k = mu_q.size(1)
        mu_diff = mu_p - mu_q
        mu_diff_sq = torch.mul(mu_diff, mu_diff)
        logdet_std_q = torch.sum(2 * torch.log(torch.clamp(std_q, min=1e-8)), dim=1)
        logdet_std_p = torch.sum(2 * torch.log(torch.clamp(std_p, min=1e-8)), dim=1)
        fs = torch.sum(torch.div(std_q ** 2, std_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, std_p ** 2), dim=1)
        kl_divergence = (fs - k + logdet_std_p - logdet_std_q)*0.5
        return kl_divergence.mean()

    def reparameterize(self, mu, std):
        batch_size = mu.shape[0]
        z = torch.randn(self.sample_size, batch_size, mu.shape[1]).to(device)
        return mu + std * z

    def get_logits(self, z, mu, sampling_type):
        if sampling_type == "iid":
            logits = self.classifier(z)
            mean_logits = logits.mean(dim=0)
            logits = logits.permute(1, 2, 0)
        else:
            mean_logits = self.classifier(mu)
            logits = mean_logits
        return logits, mean_logits


    def sampled_loss(self, logits, mean_logits, labels, sampling_type):
        if sampling_type == "iid":
            # During the training, computes the loss with the sampled embeddings.
            loss_fct = CrossEntropyLoss(reduce=False)
            loss = loss_fct(logits, labels[:, None].expand(-1, self.sample_size))
            loss = torch.mean(loss, dim=-1)
            loss = torch.mean(loss, dim=0)
        else:
            # During test time, uses the average value for prediction.
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(mean_logits, labels)
        return loss

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        labels=None,
        sampling_type="iid",
        epoch=1,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            A classification loss is computed (Cross-Entropy).
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.LongformerConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Examples::
        from transformers import LongformerTokenizer, LongformerForSequenceClassification
        import torch
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
        """

        final_outputs = {}
        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )
        pooled_output = outputs[0] # last_hidden_state 
        pooled_output = pooled_output[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        loss = {}

        pooled_output = self.mlp(pooled_output)
        batch_size = pooled_output.shape[0]
        mu, std = self.estimate(pooled_output, self.emb2mu, self.emb2std)
        mu_p = self.mu_p.view(1, -1).expand(batch_size, -1)
        std_p = torch.nn.functional.softplus(self.std_p.view(1, -1).expand(batch_size, -1))
        kl_loss = self.kl_div(mu, std, mu_p, std_p)
        z = self.reparameterize(mu, std)
        final_outputs["z"] = mu

        if self.kl_annealing == "linear":
            beta = min(1.0, epoch*self.beta)
                
        sampled_logits, logits = self.get_logits(z, mu, sampling_type)
        if labels is not None:
            ce_loss = self.sampled_loss(sampled_logits, logits, labels.view(-1), sampling_type)
            loss["loss"] = ce_loss + (beta if self.kl_annealing == "linear" else self.beta) * kl_loss
    
        final_outputs.update({"logits": logits, "loss": loss, "hidden_attention": outputs[2:]})
        return final_outputs

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

def train_model(model, train_loader, val_loader, lr, num_epochs=5, validate_after_every_epoch=False, early_stop=False):
    early_stopper = EarlyStopper(patience=1, min_delta=0.01)
    # Set the optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    num_train_steps = len(train_loader) * num_epochs
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    model.zero_grad()
    # Fine-tune the model
    for epoch in tqdm(range(num_epochs), position=0, leave=True):
        model.train()
        predictions = []
        true_labels = []
        for batch in (tqdm(train_loader, position=0, leave=True, desc='Training')):
            # Retrieve input ids, attention mask, and labels from the batch
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            # Create a global attention mask
            temp_list = [0] * (len(input_ids[0]) - 1)
            global_attention_mask = [1] + temp_list
            global_attention_mask = torch.from_numpy(np.array(((len(input_ids)))*[global_attention_mask])).to(device)
            
            # Pass inputs through the model and calculate the loss
            outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, labels=labels, epoch=epoch)
            loss = outputs["loss"]["loss"]
            logits = outputs["logits"]

            # Record predictions and actual labels for accuracy
            batch_predictions = torch.argmax(logits, dim=1)
            predictions += batch_predictions.cpu().numpy().tolist()
            true_labels += labels.cpu().numpy().tolist()

            # Zero the gradients, backpropagate, and update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            model.zero_grad()
            gc.collect()

        if validate_after_every_epoch:
            _, val_loss = validation_metrics(model, val_loader, epoch)
        elif epoch % 2 == 0:
            _, val_loss = validation_metrics(model, val_loader, epoch)
        
        # Early Stop to prevent overfitting
        if epoch % 2 and early_stop and early_stopper.early_stop(val_loss):    
            print('EARLY STOP')         
            break

    return accuracy_score(true_labels, predictions)

def validation_metrics(model, val_loader, epoch=None):
    model.eval()
    total_loss = 0
    total = 0
    predictions = []
    true_labels = []
    for batch in tqdm(val_loader, position=0, leave=True, desc='Validation'):
        # Retrieve input ids, attention mask, and labels from the batch
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        # Create a global attention mask
        temp_list = [0] * (len(input_ids[0]) - 1)
        global_attention_mask = [1] + temp_list
        global_attention_mask = torch.from_numpy(np.array(((len(input_ids)))*[global_attention_mask])).to(device)

        # Compute model outputs and loss for the current batch
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, labels=labels, sampling_type='argmax')
            loss = outputs["loss"]["loss"]
            logits = outputs["logits"]
            batch_predictions = torch.argmax(logits, dim=1)

        total_loss += loss.item() * input_ids.size(0)
        total += labels.shape[0]
        predictions += batch_predictions.cpu().numpy().tolist()
        true_labels += labels.cpu().numpy().tolist()

    # Compute average loss and accuracy across all batches
    avg_loss = total_loss / (total)
    accuracy = accuracy_score(true_labels, predictions)

    # Print the validation metrics
    if(epoch is not None):
      result = f'Epoch {epoch + 1}: Validation Loss={avg_loss:.4f}, Validation Accuracy={accuracy:.4f}'
    else:
      result = f'Validation Loss={avg_loss:.4f}, Validation Accuracy={accuracy:.4f}'
    print(result)

    return accuracy, avg_loss

def test_model(model, test_loader):
    model.eval()
    total_loss = 0
    total = 0
    predictions = []
    true_labels = []
    for batch in tqdm(test_loader, position=0, leave=True, desc='Testing'):
        # Retrieve input ids, attention mask, and labels from the batch
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        # Create a global attention mask
        temp_list = [0] * (len(input_ids[0]) - 1)
        global_attention_mask = [1] + temp_list
        global_attention_mask = torch.from_numpy(np.array(((len(input_ids)))*[global_attention_mask])).to(device)
        
        # Compute model outputs and loss for the current batch
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, labels=labels, sampling_type='argmax')
            loss = outputs["loss"]["loss"]
            logits = outputs["logits"]
            batch_predictions = torch.argmax(logits, dim=1)

        
        total_loss += loss.item() * input_ids.size(0)
        total += labels.shape[0]
        predictions += batch_predictions.cpu().numpy().tolist()
        true_labels += labels.cpu().numpy().tolist()

    # Compute average loss, accuracy and mcc across all batches
    avg_loss = total_loss / (total)
    accuracy = accuracy_score(true_labels, predictions)
    f1Score = f1_score(true_labels, predictions)
    mcc_score = mcc(y_pred=predictions, y_true=true_labels)
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    
    print(f'Testing Model : Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}, MCC={mcc_score:.4f}, F1 Score={f1Score:.4f}')
    print(f'True Positives: {tp}: True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}')
    return accuracy

def set_seed(seed):
    # Set the seed values for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def getArgs():
    # Add arguments to the parser object
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, required=True, help="The dataset to use", choices={"2", "3", "combined"}, default="2")
    parser.add_argument('--max_seq_length', type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded", default=1024)
    parser.add_argument('--num_epochs', type=int, help="The number of epochs to train for", default=5)
    parser.add_argument('--batch_size', type=int, help="The batch size to use", default=4)
    parser.add_argument('--ib_dim', type=int, help="Specifies the dimension of the information bottleneck", default=384)
    parser.add_argument('--optuna', help="Run optuna to find best hyperparameters", action='store_true')
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--activation", help="The activation function to use", type=str, choices=["sigmoid", "relu"], default="relu")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate")
    parser.add_argument('--validate_after_every_epoch', help="Runs validation after every epoch", action='store_true')
    parser.add_argument('--early_stop', help="Use early stopping in model training", action='store_true')    
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--num_optuna_trials", type=int, default=20, help="Number of trials to run for Optuna")
    parser.add_argument("--out_dir", type=str, default="vilongformer_output", help="The output directory where the model will be stored")

    # Parse the arguments
    args = parser.parse_args()

    # Set a seed for reproducibility
    set_seed(args.seed)

    return args

# Callback function for optuna to keep track of the best accuracy so far
def callback(study, trial):
    if study.best_trial.number == trial.number:
        current_acc = trial.user_attrs["current_acc"]
        study.set_user_attr(key="best_acc", value=current_acc)

# Function to free up memory
def garbage(study, trial):
    # Call garbage collector to free up unused memory
	gc.collect()

# Objective function used by optuna to find the best set of hyperparameters
def objective(trial, study, train_loader, val_loader):
    args = getArgs()

    # Set up trial values to experiment with
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    dropout_val = trial.suggest_float("dropout_val", 0, 0.5, step=0.1)
    ib_dim = trial.suggest_int("ib_dim", 200, 500, step=5)
    activation = trial.suggest_categorical("activation", ["relu", "sigmoid"])
    # epoc = trial.suggest_int("epochs", 5, 50, 5)

    config = getConfig()
    config.hidden_dropout_prob = dropout_val
    config.activation = activation
    config.ib_dim = ib_dim

    # Train model and get validation accuracy score
    model = ViLongformer.from_pretrained('allenai/longformer-base-4096', config=config)
    model.to(device)
    train_acc = train_model(model, num_epochs=3, early_stop=args.early_stop, lr=lr, train_loader=train_loader, val_loader=val_loader)
    val_acc, _ = validation_metrics(model, val_loader)

    # Set values to keep track of best accuracy so far
    trial.set_user_attr(key="current_acc", value=val_acc)
    trial.set_user_attr(key="train_acc", value=train_acc)
    best_acc = study.user_attrs["best_acc"]

    # If current accuracy is better than the best, save the model
    if(val_acc > best_acc):
        if not os.path.exists(os.getcwd() + '/vilongformer_models/'):
            os.makedirs(os.getcwd() + '/vilongformer_models/')
        with open(os.getcwd() + "/vilongformer_models/vi-longformer-subset{}-{}.pt".format(args.subset, trial.number), "wb") as fout:
            torch.save(model, fout)
        
    return val_acc

def main():
    args = getArgs()

    # Get the dataset
    dataset = datasets[args.subset]

    # Read the train, test and val splits
    df_train = pd.read_csv('./' + dataset + '_train.csv')
    df_test = pd.read_csv('./' + dataset + '_test.csv')
    df_val = pd.read_csv('./' + dataset + '_val.csv')

    train_loader = prepare_data(df_train, args.batch_size)
    test_loader = prepare_data(df_test, args.batch_size)
    val_loader = prepare_data(df_val, args.batch_size)

    # If optuna flag is enabled perform hyperparameter optimization
    if(args.optuna):
        # Create optuna study
        STUDY_NAME = "vi-longformer-subset" + str(args.subset)
        study = optuna.create_study(direction='maximize', study_name=STUDY_NAME, storage='sqlite:///' + STUDY_NAME + '.db', load_if_exists=True)
        
        # Initialzie best_acc var to 0
        if(study.user_attrs.get('best_acc') is None):
            study.set_user_attr("best_acc", 0)
        
        # Create a wrapper function to be able to pass more parameters to the objective function
        func = lambda trial: objective(trial, study, train_loader, val_loader)

        # If trials have already been conducted, print the current best and total number of trials
        if(len(study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))) > 1):
            print(study.best_trial)
            print('Total number of trials', len(study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))))

        # Enque first trail
        study.enqueue_trial({"lr": 2e-5, "dropout_val": 0.2, "activation": "sigmoid", "ib_dim": 384}, skip_if_exists=True)

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
        model = torch.load(os.getcwd() + '/vilongformer_models/' + STUDY_NAME + '-' + str(study.best_trial.number) + '.pt')
        print('TESTING MODE')
        test_model(model, test_loader)
    else:
        # Initialize model, then train and test
        config = getConfig()
        model = ViLongformer.from_pretrained('allenai/longformer-base-4096', config=config)
        model.to(device)

        train_model(model, train_loader, val_loader, args.learning_rate, num_epochs=args.num_epochs, validate_after_every_epoch=args.validate_after_every_epoch, early_stop=args.early_stop)
        test_model(model, test_loader)
        if not os.path.exists(os.getcwd() + "/" + args.out_dir):
            os.makedirs(os.getcwd() + "/" + args.out_dir)
        with open(os.getcwd() + "/" + args.out_dir + "/vilongformer-subset{}.pt".format(args.subset), "wb") as fout:
            torch.save(model, fout)

if __name__ == "__main__":
   with warnings.catch_warnings(record=True):
    main()