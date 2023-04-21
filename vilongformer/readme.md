# VILongformer

This script trains a PyTorch model using the ViLongformer architecture on a given dataset.

## Python requirements
This code was tested on the UoM CSF with the following:

* PyTorch (version == 1.10.2)
* Transformers (version == 4.18.0)
* Python (version == 3.6.5)
* Tqdm (version == 4.64.1)
* Scikit-learn (version == 0.24.1)
* Optuna (version == 2.10.1)

## Datasets

The datasets have been included in this file, they are 'subset2.csv', 'subset3.csv', and 'combined.csv'. Alongside this I have included the train/test splits for each which are needed to run the code.

**Please move the python script inside the datasets folder ensuring that the python script and the datasets are in the same directory before running**

## Parameters in the code

Most of the arguments are optional and have defaults set, they can be changed/enabled as neccessary. Subset is the only parameter that is ***not optional***.

* `subset` ***Required*** Specifcies which subset to use *Choices {2, 3, combined}* 
* `out_dir` The output directory where the model will be stored  *Default set to vilongformer_output*
* `max_seq_length ` Specifies the maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded *Default set to 1024*
* `num_epochs` The number of epochs to train for *Default set to 5*
* `batch_size` The batch size to use *Default set to 4*
* `ib_dim` Specifies the dimension of the information bottleneck *Default set to 384*
* `learning_rate` The initial learning rate for Adam *Default set to 2e-5*
* `activation` The activation function to use *Default set to relu - Choices {relu, sigmoid}*
* `dropout` The dropout rate *Default set to None*
* `seed` The random seed for initialization *Default set to 42*
* `num_optuna_trials` Number of trials to run for Optuna *Default set to 20*
* `early_stop` If this flag is enabled use early stopping in model training. Used in normal training and optuna fine tuninng. 
* `optuna` If this flag is enabled, optuna trials are run to find the best hyperparameters
* `validate_after_every_epoch` If this flag is enabled, Runs validation after every epoch

## Usage
An example of running the training script to train VILongformer :

```
python3 vilongformer.py --subset 2
```

An example of running the training script to train VILongformer with optuna hyperparameter optimization :

```
python3 vilongformer.py --subset 2 --optuna
```