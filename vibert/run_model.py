""" Finetuning the library models for sequence classification on AITA (VIBERT, VIRoBERTa, VIALBERT)."""

import argparse
import logging
import os
import random
import pandas as pd
import warnings
import optuna

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import gc

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertTokenizer,
    BertConfig,
    BertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    
    get_linear_schedule_with_warmup,
)
from vibert import BertForSequenceClassification, RobertaForSequenceClassification, AlbertForSequenceClassification
from vibert_data import convert_examples_to_features, processors, output_modes, get_metrics, get_confusion_matrix

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
}

# The models to be used with the VI setting
AVAILABLE_MODELS = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "albert": "albert-base-v2",
}

# The two available datasets
DATASETS = {
   "2": "subset2",
   "3": "subset3",
   "combined": "combined"
}

def set_seed(args):
    # Set the seed values for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)

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
    
def train(args, train_dataset, model, tokenizer, early_stop=False):
    """ Train the model """
    stop_training = 0
    early_stopper = EarlyStopper(patience=1, min_delta=0.01)
    def save_model(args, global_step, model, optimizer, scheduler, tokenizer):
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", output_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]},
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,\
                                                num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    for epoch in train_iterator:
        if stop_training == 2:
            break;
        epoch = epoch + 1
        args.epoch = epoch
        epoch_iterator = tqdm(train_dataloader, desc="Iteration " + str(epoch), position=0, leave=True, disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            if(args.model_type == 'roberta'):
              inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
            else:
              inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            
            inputs["token_type_ids"] = (
                batch[2] if args.model_type in ["bert", "albert"] else None
            )  #RoBERTa don't use segment_ids
            outputs = model(**inputs, epoch=epoch)
            loss = outputs["loss"]["loss"]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
           
            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _, _, _ = evaluate(args, model, tokenizer)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    save_model(args, global_step, model, optimizer, scheduler, tokenizer)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        # Evaluates the model after each epoch.
        if args.evaluate_after_each_epoch and epoch % 2 == 0:
            results, _, _, _ = evaluate(args, model, tokenizer, epoch=epoch)
            if early_stop and early_stopper.early_stop(results['dev_loss']):
               print('EARLYSTOP')
               break
            #save_model(args, global_step, model, optimizer, scheduler, tokenizer)

    return global_step, tr_loss / global_step, results['acc_train']

def binarize_preds(preds):
    # maps the third label (neutral one) to first, which is contradiction.
    preds[preds == 2] = 0
    return preds

def compute_metrics(preds, out_label_ids):
    return get_metrics(preds, out_label_ids)

def evaluate(args, model, tokenizer, prefix="", sampling_type="argmax", save_results=True, epoch=0):
    results = {}
    all_preds = {}
    all_zs = {}
    all_labels = {}
    for eval_task in args.eval_tasks:
        for eval_type in args.eval_types:
            print("Evaluating on "+eval_task+" with eval_type ", eval_type)
            eval_dataset, num_classes = load_and_cache_examples(args, eval_task, tokenizer, eval_type)
            args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
            # Note that DistributedSampler samples randomly
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

            # multi-gpu eval
            if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
                model = torch.nn.DataParallel(model)

            # Eval!
            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            out_label_ids = None
            zs = []
            for batch in tqdm(eval_dataloader, desc="Evaluating", position=0, leave=True): # ameer made change
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)

                with torch.no_grad():
                    if(args.model_type == 'roberta'):
                        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
                    else:
                        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                    
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "albert"] else None
                    )  #RoBERTa don't use segment_ids

                    no_label = False
                    if no_label:
                        inputs["labels"] = None
                    outputs = model(**inputs, sampling_type=sampling_type)
                    tmp_eval_loss, logits = outputs["loss"], outputs["logits"]

                    eval_loss += tmp_eval_loss['loss'].mean().item()
                nb_eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = None if no_label else inputs["labels"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = None if no_label else np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

                zs.append(outputs["z"])

            eval_loss = eval_loss / nb_eval_steps
            if args.output_mode == "classification":
                preds = np.argmax(preds, axis=1)

                # binarize the labels and predictions if needed.
                if num_classes == 2 and args.binarize_eval:
                    preds = binarize_preds(preds)
                    out_label_ids = binarize_preds(out_label_ids)

            elif args.output_mode == "regression":
                preds = np.squeeze(preds)

            all_preds[eval_task + "_" + eval_type] = preds
            all_zs[eval_task+"_"+eval_type] = torch.cat(zs)
            all_labels[eval_task+"_"+eval_type] = out_label_ids

            no_label = False
            if not no_label:
                temp = compute_metrics(preds, out_label_ids)
                if len(args.eval_tasks) > 1:
                    # then this is for transfer and we need to know the name of the datasets.
                    temp = {eval_task+"_"+k + '_' + eval_type: v for k, v in temp.items()}
                else:
                    temp = {k + '_' + eval_type: v for k, v in temp.items()}
                    temp[eval_type + '_loss'] = eval_loss
                results.update(temp)
        if args.eval_types == ['test']:
            tn, fp, fn, tp = get_confusion_matrix(preds, out_label_ids)
            print(f'Testing Model : Loss={results["test_loss"]:.4f}, Accuracy={results["acc_test"]:.4f}, MCC={results["mcc_test"]:.4f}, F1 Score={results["f1Score_test"]:.4f}')
            print(f'True Positives: {tp}: True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}')
        elif args.eval_types == ['dev']:
            print(f'Val Loss={results["dev_loss"]:.4f}, Val Accuracy={results["acc_dev"]:.4f}')
        else:
            print(f'Train Loss={results["train_loss"]:.4f}, Train Accuracy={results["acc_train"]:.4f}, Val Loss={results["dev_loss"]:.4f}, Val Accuracy={results["acc_dev"]:.4f}')
            
    return results, all_preds, all_zs, all_labels

def load_and_cache_examples(args, task, tokenizer, eval_type):
    
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Get the dataset
    dataset = DATASETS[args.subset]
    
    processor = processors[task]()
    output_mode = output_modes[task]
    label_list = processor.get_labels()

    # Read the train, test and val splits
    df_train = pd.read_csv('./' + dataset + '_train.csv')
    df_test = pd.read_csv('./' + dataset + '_test.csv')
    df_val = pd.read_csv('./' + dataset + '_val.csv')

    if eval_type == "train":
        examples = (processor.get_train_examples(df_train))
    elif eval_type == "test":
        examples = (processor.get_dev_examples(df_test))
    elif eval_type == "dev":
        examples = (processor.get_validation_examples(df_val))

    features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=label_list,
        max_length=args.max_seq_length,
        pad_on_left=False, 
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
        output_mode=output_mode,
        no_label=False,
        model_type=args.model_type if (args.model_type == 'roberta') else None
    )
    
    if args.local_rank == 0 and eval_type == "train":
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    if(args.model_type != 'roberta'):
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    else:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)
    return dataset, processor.num_classes

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout", type=float, default=None, help="dropout rate.")
    parser.add_argument("--kl_annealing", choices=[None, "linear"], default="linear")
    parser.add_argument("--evaluate_after_each_epoch", action="store_true", help="Evaluates the model after\
            each epoch and saves the best model.")
    parser.add_argument("--activation", type=str, choices=["tanh", "sigmoid", "relu"], \
                        default="relu")
    parser.add_argument("--eval_types", nargs="+", type=str, default=["dev", "train"], \
                        choices=["train", "test", "dev"], help="Specifies the types to evaluate on,\
                            can be dev, test, train.")
    parser.add_argument("--binarize_eval", action="store_true", help="If specified, binarize the predictions, and\
            labels during the evaluations in case of binary-class datasets.")
    # Ib parameters.
    parser.add_argument("--beta", type=float, default=1.0, help="Defines the weight for the information bottleneck\
            loss.")
    parser.add_argument("--sample_size", type=int, default=5, help="Defines the number of samples for the ib method.")
    parser.add_argument("--ib_dim", default=128, type=int,
                        help="Specifies the dimension of the information bottleneck.")

    parser.add_argument("--eval_tasks", nargs="+", default=[], type=str, help="Specifies a list of evaluation tasks.")
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="aita",
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=0, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=0, help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--data_seed", type=int, default=66, help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--subset', type=str, required=True, help="The dataset to use", choices={"2", "3", "combined"}, default="2")
    parser.add_argument('--optuna', help="run optuna to find best hyperparams", action='store_true')
    parser.add_argument("--num_optuna_trials", type=int, default=20, help="Number of trials to run for Optuna")
    args = parser.parse_args()
    if args.evaluate_after_each_epoch and "dev" not in args.eval_types:
        args.eval_types.append("dev")

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    
    args.model_name_or_path = AVAILABLE_MODELS[args.model_type]
    args.ib = True
    args.do_eval = True
    args.do_train = True

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    if len(args.eval_tasks) == 0:
        args.eval_tasks = [args.task_name]

    args.task_name = args.task_name.lower()
    return args

# Objective function used by optuna to find the best set of hyperparameters
def objective(trial, study):
    args = get_args()

    # Set up trial values to experiment with
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    ib_dim = trial.suggest_int("ib_dim", 200, 500, step=5)
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    beta = trial.suggest_float("beta", 1e-5, 1)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    activation = trial.suggest_categorical("activation", ["relu", "sigmoid"])
    args.train_batch_size = batch_size
    args.learning_rate = lr

    
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    config.activation = activation
    config.sample_size = args.sample_size
    config.kl_annealing = args.kl_annealing
    config.ib = args.ib
    
    # gleu_new which is the default for albert causes pickle errors with torch.save
    if(args.model_type == 'albert'):
        config.hidden_act = 'gelu'
    
    config.hidden_dropout_prob = dropout
    config.ib_dim = ib_dim
    config.hidden_dim = (768 + ib_dim) // 2
    config.beta = beta

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

    # Training
    if args.do_train:
        train_dataset, _ = load_and_cache_examples(args, args.task_name, tokenizer, eval_type="train")
        global_step, tr_loss, tr_acc = train(args, train_dataset, model, tokenizer, early_stop=True)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        trial.set_user_attr(key="train_acc", value=tr_acc)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            args.eval_types = ['dev']
            result, _, _, _ = evaluate(args, model, tokenizer, prefix=prefix)
            trial.set_user_attr(key="current_acc", value=result['acc_dev'])
            best_acc = study.user_attrs["best_acc"]

            if(result['acc_dev'] > best_acc):
                print(args.model_type, args.subset, trial.number)
                if not os.path.exists(os.getcwd() + '/vi_models/'):
                    os.makedirs(os.getcwd() + '/vi_models/')
                with open(os.getcwd() + "/vi_models/vi-{}-subset{}-{}.pt".format(args.model_type, args.subset, trial.number), "wb") as fout:
                    torch.save(model, fout)
            return result['acc_dev']

# Function to free up memory
def garbage(study, trial):
	gc.collect()

# Callback function for optuna to keep track of the best accuracy so far
def callback(study, trial):
    if study.best_trial.number == trial.number:
        current_acc = trial.user_attrs["current_acc"]
        study.set_user_attr(key="best_acc", value=current_acc)

def main():
    args = get_args()

    # If optuna flag is enabled perform hyperparameter optimization
    if args.optuna:
        # Create optuna study
        STUDY_NAME = 'vi-' + args.model_type + '-subset' + str(args.subset)
        study = optuna.create_study(study_name=STUDY_NAME, direction='maximize', storage='sqlite:///' + STUDY_NAME + '.db', load_if_exists=True)
      
        # Initialzie best_acc var to 0
        if(study.user_attrs.get('best_acc') is None):
            study.set_user_attr("best_acc", 0)

        # If trials have already been conducted, print the current best and total number of trials
        if(len(study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))) > 1):
            print(study.best_trial)
            print('Total number of trials', len(study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))))

        # Create a wrapper function to be able to pass more parameters to the objective function
        func = lambda trial: objective(trial, study)

        # Enque first trail
        study.enqueue_trial({"batch_size": 16, "lr": 2e-5, "ib_dim": 384, "dropout": 0.2, "beta": 1e-05, "activation": "sigmoid"}, skip_if_exists=True)

        # Start Optuna Optimization
        study.optimize(func, n_trials=args.num_optuna_trials ,show_progress_bar=True, callbacks=[callback, garbage])
        
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
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        best_model = torch.load(os.getcwd() + '/vi_models/' + STUDY_NAME + '-' + str(study.best_trial.number) + '.pt')

        # Run Testing
        results = {}
        
        args.eval_batch_size = trial.params['batch_size']
        args.ib_dim = trial.params['ib_dim']
        args.beta = trial.params['beta']
        best_model.to(args.device)
        args.output_mode = output_modes[args.task_name]
        args.eval_types = ['test']
        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        result, _, _, _ = evaluate(args, best_model, tokenizer)
        results.update(result)
        print(args.model_type)
        return results
    else:
        # Initialize model, then train and test
        processor = processors[args.task_name]()
        args.output_mode = output_modes[args.task_name]

        label_list = processor.get_labels()
        num_labels = len(label_list)

        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        args.model_type = args.model_type.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        if args.model_type in ["bert", "roberta", "albert"]:
            # bert dim is 768.
             args.hidden_dim = (768 + args.ib_dim) // 2
        # sets the parameters of IB or MLP baseline.
        config.ib = args.ib
        config.activation = args.activation
        config.hidden_dim = args.hidden_dim
        config.ib_dim = args.ib_dim
        config.beta = args.beta
        config.sample_size = args.sample_size
        config.kl_annealing = args.kl_annealing
        if args.dropout is not None:
            config.hidden_dropout_prob = args.dropout

        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model.to(args.device)

        logger.info("Training/evaluation parameters %s", args)

        if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            # Create output directory if needed
            if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(args.output_dir)

    
        # Training
        if args.do_train:
            train_dataset, _ = load_and_cache_examples(args, args.task_name, tokenizer, eval_type="train")
            global_step, tr_loss, _ = train(args, train_dataset, model, tokenizer, early_stop=True)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            logger.info("Saving model checkpoint to %s", args.output_dir)
            #  Save a trained model, configuration and tokenizer using `save_pretrained()`.
            #  They can then be reloaded using `from_pretrained()`
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            #  Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

            #  Load a trained model and vocabulary that you have fine-tuned
            model = model_class.from_pretrained(args.output_dir)
            tokenizer = tokenizer_class.from_pretrained(args.output_dir)
            model.to(args.device)

        # Evaluation
        print('EVALUATION')
        results = {}

        if args.do_eval and args.local_rank in [-1, 0]:
            tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
            checkpoints = [args.output_dir]
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
                model = model_class.from_pretrained(checkpoint)
                model.to(args.device)
                args.eval_types = ['test']
                result, _, _, _ = evaluate(args, model, tokenizer, prefix=prefix)
                results.update(result)
        return results


if __name__ == "__main__":
    with warnings.catch_warnings(record=True):
      main()
