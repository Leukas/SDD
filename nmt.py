# nmt.py
import os

from utils.sampling import prep_batch_for_collating
import transformers
from transformers import T5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
from transformers.trainer_callback import PrinterCallback
from utils.byt5_tokenizer_new import ByT5Tokenizer
import datasets
from datasets import load_dataset
from functools import partial
from types import MethodType
from utils.ds_tokenizer import DSTokenizer
from utils.utils import get_checkpoint_path, get_reload_path, tokenize, padding_collate_fn, load_model
from utils.metrics import compute_bleu
from utils.logging_utils import load_logger, LogFlushCallback
from nmt.nmt_eval import evaluation_loop
from emb_analysis.emb_training import EmbeddingLengthSaveCallback

import torch
import argparse
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger = load_logger(logger)
datasets.logging.set_verbosity(logging.NOTSET) # patch to remove tqdm bars from dataloading

parser = argparse.ArgumentParser("Fine-tuning NMT")
# Model args
parser.add_argument("--model_type", type=str, help="The model type to use. \
                    Currently supported (case insensitive): \
                    {word, char, fixed, buffered_fixed, twostep, twostep_word}" )
parser.add_argument("--diff_pretrained_type", type=str, default="", help="Model type of reloaded model, if different.")
parser.add_argument("--reload_id", type=str, default="", help="Job ID of model to reload.")
parser.add_argument("--reload_best", action="store_true", help="Reload best model. Done automatically for evaluation, \
                    but for training the default is the last.")
parser.add_argument("--tiny", action="store_true", help="Use a tiny model, \
                    with 3 layers.")
parser.add_argument("--emb_dim", type=int, default=512, help="Size of embedding dimension.")
parser.add_argument("--max_pos_embs", type=int, default=1024, help="Size of embedding dimension.")
parser.add_argument("--ds_factor", type=int, default=4, help="Downsampling factor for gbst and full")
# Data args
parser.add_argument("--dataset", type=str, default="iwslt2017", help="Dataset used.")
parser.add_argument("--langs", type=str, default="de,en", help="Languages used, comma-separated.")
parser.add_argument("--spm_model", type=str, default="", help="Path of sentencepiece model. Only relevant for word models.")
parser.add_argument("--data_lim", type=int, default=0, help="Limit number of sentences to use, or 0 to disable.")
parser.add_argument("--cache_dir", type=str, default=None, help="Directory to cache data in. Set to /dev/shm/ for faster loading")
parser.add_argument("--bos_buffer", type=int, default=1, help="Number of BOS tokens to pad on the beginning. \
                    Already included for full GBST model. Only relevant for specific experimental models.")
parser.add_argument("--recache_tok", action="store_true", help="Recaches the tokenization, \
                    needed if tokenization method changes.")
parser.add_argument("--filter_both_langs", action="store_true", help="Filter both language sides")
# Training args
parser.add_argument("--debug", action="store_true", help="Activates debug mode, \
                    which shortens steps until evaluation, and enables tqdm.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
parser.add_argument("--batch_by_tokens", type=int, default=0, help="Batch by number of tokens, batch_size ignored if set.")
parser.add_argument("--grad_acc", type=int, default=4, help="Accumulate gradients.")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
parser.add_argument("--eval_steps", type=int, default=5000, help="Number of steps between evaluation.")
parser.add_argument("--save_steps", type=int, default=5000, help="Number of steps between saving.")
parser.add_argument("--logging_steps", type=int, default=100, help="Number of steps between logging.")
parser.add_argument("--log_embedding_changes", action="store_true", help="Log the average change in embeddings.")
parser.add_argument("--fp16", action="store_true", help="Use fp16 precision.")
parser.add_argument("--eval_only", action="store_true", help="Only evaluate.")
parser.add_argument("--resume", action="store_true", help="Resume training.")
parser.add_argument("--dont_group_by_length", action="store_true", help="Dont group by length")

if __name__ == "__main__":
    datasets.utils.logging.set_verbosity_error()
    args = parser.parse_args()
    args.task = "nmt"
    args.model_type = args.model_type.lower()
    args.langs = args.langs.split(",")
    args.reload_best = args.reload_best or args.eval_only
    logger.info(args)

    char = True
    if "word" in args.model_type:
        char = False

    if args.debug:
        args.logging_steps = 1
        args.eval_steps = 10

        
    pretrained = 'google/byt5-small' if char else 't5-small'
    if args.model_type == "ds_word":
        folder = "data/iwslt2017/"
        files = [folder + "train.de", folder + "train.en"]
        tok = DSTokenizer(files, ds_factor=args.ds_factor)
    else:
        tok = ByT5Tokenizer.from_pretrained(pretrained) if char else T5Tokenizer.from_pretrained(pretrained)
    if args.spm_model:
        assert "word" in args.model_type
        tok = T5Tokenizer(args.spm_model, model_max_length=512)

    args.tok = tok
    args.tok.bos = args.tok.vocab_size - 1
    args.tok.bos_buffer = args.bos_buffer
    args.tok.langs = args.langs

    pretrained_model = pretrained
    args.save_path, reload_id = get_reload_path(args)
    args.reload_path = args.save_path if args.reload_id else None
    args.checkpoint_path = get_checkpoint_path(args.reload_path, False) if args.resume else None

    model = load_model(pretrained_model, args)
    logger.info("Model loaded.")

    def get_num_params(model):
        total = 0
        for param in model.parameters():
            total += param.numel()
        return total

    logger.info("----- Model Parameters -----")
    logger.info("Total: %d" % get_num_params(model))
    # logger.info("Enc Embeds: %d" % get_num_params(model.encoder.embeddings))
    # logger.info("Dec Embeds: %d" % get_num_params(model.decoder.bert.embeddings))
    logger.info("----------------------------")

    model.config.bos_token_id = args.tok.bos
    model.config.eos_token_id = args.tok.eos_token_id

    # dataset = load_dataset('dataloading/translation_dataset.py', 'de-en')
    if args.dataset == "iwslt2017":
        dataset = load_dataset('dataloading/iwslt2017_dataset.py', 'iwslt2017-%s-%s' % tuple(args.langs), cache_dir=args.cache_dir)
    elif args.dataset == "wmt14":
        assert "en" in args.langs and "de" in args.langs
        dataset = load_dataset('dataloading/wmt14_deen.py', '%s-%s' % tuple(args.langs), cache_dir=args.cache_dir)
    elif args.dataset == "biomed":
        assert "en" in args.langs and "de" in args.langs
        dataset = load_dataset('dataloading/translation_biomed.py', '%s-%s' % tuple(args.langs), cache_dir=args.cache_dir)
    elif args.dataset == "flores":
        dataset = load_dataset('dataloading/translation_flores.py', '%s-%s' % tuple(args.langs), cache_dir=args.cache_dir)
    elif args.dataset == "entr":
        assert "en" in args.langs and "tr" in args.langs
        dataset = load_dataset('dataloading/translation_entr.py', 'en-tr', cache_dir=args.cache_dir)
        # dataset = load_from_disk('data/wmt14/train', keep_in_memory=True)
    elif args.dataset == "enfi":
        assert "en" in args.langs and "fi" in args.langs
        dataset = load_dataset('dataloading/translation_enfi.py', 'en-fi', cache_dir=args.cache_dir)
    elif args.dataset == "xhzu":
        assert "xh" in args.langs and "zu" in args.langs
        dataset = load_dataset('dataloading/translation_xhzu.py', 'xh-zu', cache_dir=args.cache_dir)
    else:
       dataset = load_dataset(args.dataset, '%s-%s' % tuple(args.langs), cache_dir=args.cache_dir)

    # if args.debug:
    #     dataset['train'] = dataset['train'].filter(lambda example, idx: idx < 1024, with_indices=True)
    #     dataset['validation'] = dataset['validation'].filter(lambda example, idx: idx < 1024, with_indices=True)
    #     dataset['test'] = dataset['test'].filter(lambda example, idx: idx < 1024, with_indices=True)

    # filter out sentence pairs where english side (or src side) is longer than 256 chars (allows for larger batch sizes)
    if not args.eval_only:
        primary_lang = 'en' if 'en' in args.langs else args.langs[0]
        dataset['train'] = dataset['train'].filter(lambda example: len(example['translation'][primary_lang]) < 256, num_proc=10)

        if args.filter_both_langs:
            secondary_lang = args.langs[1 - args.langs.index(primary_lang)]
            dataset['train'] = dataset['train'].filter(lambda example: len(example['translation'][secondary_lang]) < 256, num_proc=10)

        print("After filtering:", dataset['train'])
        print(dataset['train'][0])

    if args.data_lim: # limit amount of training data
        dataset['train'] = dataset['train'].filter(lambda example, idx: idx < args.data_lim, with_indices=True)

    if args.model_type == "buffered_fixed":
        tokenize = partial(tokenize, tokenizer=tok, buffer_ds_factor=args.ds_factor)
    else:
        tokenize = partial(tokenize, tokenizer=tok)
    dataset = dataset.map(tokenize, batched=True, num_proc=10, load_from_cache_file=not args.recache_tok)

    smooth_factor = 0.1
    if args.model_type in ['dec_only_word','dec_only','nar','nar_word','diffu','diffu_word']:
        smooth_factor = 0
    training_args = Seq2SeqTrainingArguments(args.save_path,
        evaluation_strategy="steps",
        save_strategy="steps",
        generation_max_length=512 if char else 256,
        predict_with_generate=True,
        group_by_length=args.dont_group_by_length,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        log_level="info",
        save_steps=args.save_steps,
        save_total_limit=1,
        fp16=args.fp16,
        fp16_full_eval=args.fp16,
        per_device_train_batch_size=args.batch_size if not args.batch_by_tokens else 1,
        per_device_eval_batch_size=args.batch_size if not args.batch_by_tokens else 1,
        gradient_accumulation_steps=args.grad_acc,
        metric_for_best_model="eval_bleu",
        warmup_steps=4000,
        learning_rate=args.lr,
        label_smoothing_factor=smooth_factor,
        load_best_model_at_end=True,
        resume_from_checkpoint=len(args.reload_id) > 0,
        disable_tqdm=not args.debug)

    early_stopping = EarlyStoppingCallback(early_stopping_patience=10)
    experiment_name = "%s_%s" % (args.model_type, reload_id)
    print_cb = LogFlushCallback(logger)
    # compute_metrics = partial(compute_bleu, args=args)
    cbs = [early_stopping, print_cb]
    if args.log_embedding_changes:
        emb_cb = EmbeddingLengthSaveCallback(args.model_type, logger)
        cbs.append(emb_cb)

    compute_metrics = partial(compute_bleu, args=args)

    collate_fn = padding_collate_fn if not args.batch_by_tokens \
        else partial(prep_batch_for_collating, collate_fn=padding_collate_fn)
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tok,
        args=training_args, 
        train_dataset=dataset['train'], 
        eval_dataset=dataset['validation'],
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=cbs
        )
    trainer._max_length = 1024 if char else 256
    trainer._num_beams = 1

    trainer.pop_callback(PrinterCallback)

    # modified eval loop that writes outputs to file
    trainer.evaluation_loop = MethodType(evaluation_loop, trainer)
    from nmt.nmt_gen import prediction_step
    trainer.prediction_step = MethodType(prediction_step, trainer)

    if args.batch_by_tokens:
        from utils.sampling import _get_train_sampler, _get_eval_sampler
        trainer._get_train_sampler = MethodType(_get_train_sampler, trainer)
        trainer._get_eval_sampler = MethodType(_get_eval_sampler, trainer)
        trainer.args.tokens_per_batch = args.batch_by_tokens

    if not args.eval_only:
        trainer.train(resume_from_checkpoint=args.checkpoint_path)
    
    args.eval_only = True # for writing test set to file
    if args.eval_only:
        model = model.cuda() # double-checking, some systems don't auto-port it to cuda yet
    trainer.evaluate(dataset['test'])