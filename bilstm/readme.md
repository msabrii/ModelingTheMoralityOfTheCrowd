# BiLSTM using Sentence Transformers

This script trains a PyTorch model using a custom biLSTM architecture on a given dataset.

## Python requirements
This code was tested on the UoM CSF with the following:

* PyTorch (version == 1.10.2)
* Sentence Transformers (version == 2.2.2)
* Python (version == 3.6.5)
* Scikit-learn (version == 0.24.1)
* Optuna (version == 2.10.1)
* Nltk (version == 3.6.7)

## Datasets

The datasets have been included in this file, they are 'subset2.csv', 'subset3.csv', and 'combined.csv'. Alongside this I have included the train/test splits for each which are needed to run the code.

**Please move the python script inside the datasets folder ensuring that the python script and the datasets are in the same directory before running**

## Parameters in the code

Most of the arguments are optional and have defaults set, they can be changed/enabled as neccessary. Subset is the only parameter that is ***not optional***.

* `subset` ***Required*** Specifcies which subset to use *Choices {2, 3, combined}* 
* `model` ***Required*** Specifcies which model to use to get sentence embeddings *Choices {roberta, mpnet, minilm}*
* `out_dir` The output directory where the model will be stored  *Default set to bilstm_output*
* `num_hidden_layers ` Specifies the number of hidden layers for the biLSTM *Default set to 100*
* `num_epochs` The number of epochs to train for *Default set to 5*
* `batch_size` The batch size to use *Default set to 8*
* `concatenate_hidden_states` If this flag is set LSTM will concatenate the hidden states, if this flag is not set, use pooled output
* `learning_rate` The initial learning rate for Adam *Default set to 2e-5*
* `dropout` The dropout rate *Default set to None*
* `seed` The random seed for initialization *Default set to 42*
* `early_stop` If this flag is enabled use early stopping in model training. Used in normal training and optuna fine tuninng. 
* `optuna` If this flag is enabled, optuna trials are run to find the best hyperparameters
* `num_optuna_trials` Number of trials to run for Optuna *Default set to 20*
* `validate_after_every_epoch` If this flag is enabled, Runs validation after every epoch

## Usage
An example of running the training script to train the biLSTM :

```
python3 final_bilstm.py --subset 2 --model roberta --outdir bilstm_out
```

An example of running the training script to train the biLSTM with optuna hyperparameter optimization :

```
python3 final_bilstm.py --subset 2 --model roberta --optuna
```