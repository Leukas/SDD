# nli.py
import os
import transformers
from transformers import T5Tokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
from transformers.trainer_callback import PrinterCallback
from utils.byt5_tokenizer_new import ByT5Tokenizer
import datasets
from datasets import load_dataset
from functools import partial
from types import MethodType
from utils.ds_tokenizer import DSTokenizer
from utils.utils import buffer_by_ds_factor, get_checkpoint_path, get_reload_path, load_model_encoder
from utils.logging_utils import load_logger, LogFlushCallback
from utils.metrics import compute_acc
from utils.tokenization import tokenize_nli_sdd

import torch
import argparse
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger = load_logger(logger)
datasets.logging.set_verbosity(logging.NOTSET) # patch to remove tqdm bars from dataloading

parser = argparse.ArgumentParser("Fine-tuning NLI")
# Model args
parser.add_argument("--model_type", type=str, help="The model type to use. \
                    Currently supported (case insensitive): \
                    {word, char, gbst, fixed, buffered_fixed, sdd}" )
parser.add_argument("--reload_id", type=str, default="", help="Job ID of model to reload.")
parser.add_argument("--reload_best", action="store_true", help="Reload best model. Done automatically for evaluation, \
                    but for training the default is the last.")
parser.add_argument("--tiny", action="store_true", help="Use a tiny model, \
                    with 3 layers.")
parser.add_argument("--emb_dim", type=int, default=512, help="Size of embedding dimension.")
parser.add_argument("--ds_factor", type=int, default=4, help="Downsampling factor")
# Data args
parser.add_argument("--dataset", type=str, default="snli", help="Dataset used.")
parser.add_argument("--langs", type=str, default="de,en", help="Languages used, comma-separated.")
parser.add_argument("--spm_model", type=str, default="", help="Path of sentencepiece model. Only relevant for word models.")
parser.add_argument("--data_lim", type=int, default=0, help="Limit number of sentences to use, or 0 to disable.")
parser.add_argument("--cache_dir", type=str, default=None, help="Directory to cache data in. Set to /dev/shm/ for faster loading")
parser.add_argument("--recache_tok", action="store_true", help="Recaches the tokenization, \
                    needed if tokenization method changes.")
# Training args
parser.add_argument("--debug", action="store_true", help="Activates debug mode, \
                    which shortens steps until evaluation, and enables tqdm.")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
parser.add_argument("--grad_acc", type=int, default=1, help="Accumulate gradients.")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
parser.add_argument("--eval_steps", type=int, default=5000, help="Number of steps between evaluation.")
parser.add_argument("--save_steps", type=int, default=5000, help="Number of steps between saving.")
parser.add_argument("--logging_steps", type=int, default=1000, help="Number of steps between logging.")
parser.add_argument("--eval_only", action="store_true", help="Only evaluate.")
parser.add_argument("--resume", action="store_true", help="Resume training.")

def padding_collate_fn(batch, max_len=1024):
    """ 
        Pads each list with zeros and concatenates by key.
        Input: List[{key: List[], ...}]
        Output: {key: LongTensor(), ...}
    """
    padded_batch = {}
    for key in batch[0]:
        if key == "labels":
            padded_batch[key] = torch.zeros((len(batch),1), dtype=torch.long)
        else:
            largest = min(max_len, max([len(b[key]) for b in batch]))
            padded_batch[key] = torch.zeros((len(batch), largest), dtype=torch.long)
            if "lens" in key:
                padded_batch[key] -= 100

    for i, sample in enumerate(batch):
        for key in padded_batch:
            if key == "labels":
                padded_batch[key][i] = sample[key]
            else:
                key_len = min(max_len, len(sample[key]))
                padded_batch[key][i, :key_len] = torch.LongTensor(sample[key][:key_len])

    return padded_batch

def tokenize(examples, tokenizer, ds_factor, buffer_ds_factor=0):
    encoded = {
        'input_ids': [], 
        'attention_mask': [], 
        'labels': []}

    for i in range(len(examples['premise'])):
        src = examples['premise'][i]
        tgt = examples['hypothesis'][i]

        if buffer_ds_factor:
            src = buffer_by_ds_factor(src, buffer_ds_factor)
            tgt = buffer_by_ds_factor(tgt, buffer_ds_factor)

        src_tok = tokenizer.encode(src)
        tgt_tok = tokenizer.encode(tgt)
        inp = [tokenizer.cls] * ds_factor + src_tok + [tokenizer.sep] + tgt_tok 
        encoded['input_ids'] += [inp]
        encoded['labels'] += [examples['label'][i]]

        attention_mask = torch.ones(len(inp), dtype=torch.long).tolist()
        encoded['attention_mask'] += [attention_mask]

    return encoded



if __name__ == "__main__":
    args = parser.parse_args()
    args.task = "nli"
    args.model_type = args.model_type.lower()
    args.langs = args.langs.split(",")
    args.reload_best = args.reload_best or args.eval_only
    logger.info(args)
    datasets.utils.logging.set_verbosity_error()

    char = True
    if "word" in args.model_type:
        char = False
        ds_factor = 1
    elif args.model_type == "char":
        ds_factor = 1


    if args.debug:
        args.logging_steps = 1
        args.eval_steps = 10

        
    pretrained = 'google/byt5-small' if char else 'google/mt5-small'
    if args.model_type == "ds_word":
        folder = "data/iwslt2017/"
        files = [folder + "train.de", folder + "train.en"]
        tok = DSTokenizer(files, ds_factor=args.ds_factor)
    else:
        tok = ByT5Tokenizer.from_pretrained(pretrained) if char else T5Tokenizer.from_pretrained(pretrained)
    if args.spm_model:
        assert "word" in args.model_type
        tok = T5Tokenizer(args.spm_model, model_max_length=512)

    if args.model_type == "sdd":
        args.word_tok = tok
        args.tok = ByT5Tokenizer.from_pretrained('google/byt5-small')
    else:
        args.tok = tok

    args.tok.sep = args.tok.vocab_size - 1
    args.tok.cls = args.tok.vocab_size - 2
    args.tok.langs = args.langs

    pretrained_model = pretrained
    args.save_path, reload_id = get_reload_path(args)
    args.reload_path = args.save_path if args.reload_id else None
    args.checkpoint_path = get_checkpoint_path(args.reload_path, False) if args.resume else None

    model = load_model_encoder(pretrained_model, args)

    dataset = load_dataset('snli', cache_dir=args.cache_dir, keep_in_memory=True)

    if args.debug:
        dataset['validation'] = dataset['validation'].filter(lambda example, idx: idx < 10, with_indices=True)

    # filter out instances with -1 as label (aka no label)
    dataset['train'] = dataset['train'].filter(lambda example: example['label'] != -1, num_proc=10)

    # filter out sentence pairs where english side is longer than 256 chars (allows for larger batch sizes)
    dataset['train'] = dataset['train'].filter(lambda example: len(example['premise']) + len(example['hypothesis']) + 1 < 256, num_proc=10)
    print("After filtering:", dataset['train'])

    if args.data_lim: # limit amount of training data
        dataset['train'] = dataset['train'].filter(lambda example, idx: idx < args.data_lim, with_indices=True)

    if args.model_type == "sdd":
        tokenize = partial(tokenize_nli_sdd, char_tokenizer=args.tok, word_tokenizer=args.word_tok)
    elif args.model_type == "buffered_fixed":
        tokenize = partial(tokenize, tokenizer=tok, ds_factor=args.ds_factor, buffer_ds_factor=args.ds_factor)
    else:
        tokenize = partial(tokenize, tokenizer=tok, ds_factor=args.ds_factor)
    dataset = dataset.map(tokenize, batched=True, num_proc=10, remove_columns=['premise', 'hypothesis', 'label'], load_from_cache_file=not args.recache_tok)

    experiment_name = "%s_%s" % (args.model_type, reload_id)

    training_args = TrainingArguments(args.save_path,
        evaluation_strategy="steps",
        save_strategy="steps",
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        log_level="info",
        save_steps=args.save_steps,
        save_total_limit=1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        warmup_steps=10000,
        learning_rate=args.lr,
        label_smoothing_factor=0.1,
        load_best_model_at_end=True,
        resume_from_checkpoint=len(args.reload_id) > 0,
        disable_tqdm=not args.debug)

    early_stopping = EarlyStoppingCallback(early_stopping_patience=10)

    print_cb = LogFlushCallback(logger)
    # compute_metrics = partial(compute_bleu, args=args)
    cbs = [early_stopping, print_cb]

    trainer = Trainer(
        model=model,
        tokenizer=tok,
        args=training_args, 
        train_dataset=dataset['train'], 
        eval_dataset=dataset['validation'],
        data_collator=padding_collate_fn,
        compute_metrics=compute_acc,
        callbacks=cbs
        )

    trainer.pop_callback(PrinterCallback)

    # modified eval loop that writes outputs to file
    # trainer.evaluation_loop = MethodType(evaluation_loop, trainer)

    if not args.eval_only:
        trainer.train(resume_from_checkpoint=args.checkpoint_path)
    
    args.eval_only = True # for writing test set to file
    trainer.evaluate(dataset['test'])