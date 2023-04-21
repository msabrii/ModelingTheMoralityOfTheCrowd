# VIBERT

This script trains a PyTorch model using that applies the Variational Information Bottleneck pinciple to pretrained models.

This script was made using the work of rabeehk. The github repository for their code is:
https://github.com/rabeehk/vibert

## Python requirements
This code was tested on the UoM CSF with the following :

* PyTorch (version == 1.13.1)
* Transformers (version == 2.8.0)
* Python (version == 3.10.4)
* Tqdm (version == 4.64.1)
* Scikit-learn (version == 1.2.2)
* Optuna (version == 3.1.0)

## Datasets

The datasets have been included in this file, they are 'subset2.csv', 'subset3.csv', and 'combined.csv'. Alongside this I have included the train/test splits for each which are needed to run the code.

**Please move the python script inside the datasets folder ensuring that the python script and the datasets are in the same directory before running**

## Parameters in the code

Most of the arguments are optional and have defaults set, they can be changed/enabled as neccessary. The `subset`, `model_type` and `output_dir` parameters that is ***not optional***.

* `subset` ***Required*** Specifcies which subset to use *Choices {2, 3, combined}*
* `model_type` ***Required*** Specifies which model to use to apply the VI technique to *Choices {bert, albert, roberta}*
* `output_dir` ***Required*** The output directory where the model predictions and checkpoints will be written
* `beta `Defines the weight for the information bottleneck loss *Default set to 1*
* `ib_dim` Specifies the dimension of the information bottleneck *Default set to 128*
* `sample_size` Defines the number of samples for the ib method *Default set to 5*
* `activation` The activation function to use *Default set to relu - Choices {relu, sigmoid}*
* `num_train_epochs` The number of epochs to train for *Default set to 5*
* `evaluate_after_each_epoch` If this flag is set, evaluates the model after each epoch ***This needs to be set to use early stopping***
* `per_gpu_train_batch_size` The batch size to use *Default set to 8*
* `per_gpu_eval_batch_size` The batch size to use *Default set to 8*
* `max_seq_length`The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. *Default set to 512*
* `learning_rate` The initial learning rate for Adam *Default set to 2e-5*
* `dropout` The dropout rate *Default set to None*
* `seed` The random seed for initialization *Default set to 42*
* `early_stop` If this flag is enabled use early stopping in model training. Used in normal training and optuna fine tuninng. 
* `optuna` If this flag is enabled, optuna trials are run to find the best hyperparameters
* `num_optuna_trials` Number of trials to run for Optuna *Default set to 20*
* `overwrite_output_dir` If this flag is enabled, Overwrite the content of the output directory

## Usage
An example of running the training script to train a model :

```
python run_model.py  \
    --output_dir out_dir --model_type roberta \
    --subset 2 --max_seq_length 512 --num_train_epochs 10 \
    --overwrite_output_dir --dropout 0.3 \
    --ib_dim 384 --beta 1e-05 --learning_rate 2e-5 \
    --activation relu --evaluate_after_each_epoch --seed 812 \
```

An example of running the training script to train the model with optuna hyperparameter optimization :

```
python run_model.py  \
    --output_dir out_dir --model_type roberta --optuna \
    --subset 2 --max_seq_length 512 --num_train_epochs 10 \
    --overwrite_output_dir \
    --evaluate_after_each_epoch --seed 812 \
```
