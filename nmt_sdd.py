# nmt.py
import os
from utils.sampling import prep_batch_for_collating
import transformers
from transformers import T5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
from transformers.models.bert.configuration_bert import BertConfig
from transformers.trainer_callback import PrinterCallback
from utils.byt5_tokenizer_new import ByT5Tokenizer
import datasets
from datasets import load_dataset
from functools import partial
from types import MethodType
from utils.utils import get_reload_path
from utils.metrics import compute_bleu
from utils.logging_utils import load_logger, LogFlushCallback
from nmt.nmt_eval import evaluation_loop
from utils.tokenization import padding_collate_fn_low_mem, tokenize_sdd, tokenize_sdd_to_collate
from models.lee_bert import BertWithWordLee



import torch
import argparse
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger = load_logger(logger)
datasets.logging.set_verbosity(logging.NOTSET) # patch to remove tqdm bars from dataloading
datasets.logging.get_verbosity = lambda: logging.NOTSET

parser = argparse.ArgumentParser("Fine-tuning NMT")
# Model args
parser.add_argument("--model_type", type=str, default="sdd", help="The model type to use. \
                    Currently supported (case insensitive): sdd" )
parser.add_argument("--reload_id", type=str, help="Job ID of model to reload.")
parser.add_argument("--tiny", action="store_true", help="Use a tiny model, \
                    with 3 layers.")
parser.add_argument("--emb_dim", type=int, default=512, help="Size of embedding dimension.")
parser.add_argument("--ds_factor", type=int, default=4, help="Downsampling factor for gbst and full")
parser.add_argument("--use_word_inputs", action="store_true", help="Use word inputs in addition to char inputs")
parser.add_argument("--set_max_word_len", type=int, default=0, help="Manually set max word len if needed.")
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
parser.add_argument("--low_cache_size", action="store_true", help="For when the shared memory isn't large enough to fit everything in cache.")
parser.add_argument("--filter_both_langs", action="store_true", help="Filter both language sides")
parser.add_argument("--clean_len", type=int, default=0, help="Remove sentences where a token has a byte-length of longer than X. \
                    Useful for removing languages that don't work with spm like Telugu.")
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
parser.add_argument("--fp16", action="store_true", help="Use fp16 precision.")
parser.add_argument("--eval_only", action="store_true", help="Only evaluate.")
parser.add_argument("--resume", action="store_true", help="Resume training.")
parser.add_argument("--dont_group_by_length", action="store_true", help="Dont group by length")


def padding_collate_fn(batch, max_len=1024):
    """ 
        Pads each list with zeros and concatenates by key.
        Input: List[{key: List[], ...}]
        Output: {key: LongTensor(), ...}
    """
    padded_batch = {}
    for key in batch[0]:
        largest = min(max_len, max([len(b[key]) for b in batch]))
        padded_batch[key] = torch.zeros((len(batch), largest), dtype=torch.long)
        if key == "labels" or "lens" in key:
            padded_batch[key] -= 100
    
    for i, sample in enumerate(batch):
        for key in padded_batch:
            key_len = min(max_len, len(sample[key]))
            padded_batch[key][i, :key_len] = torch.LongTensor(sample[key][:key_len])

    return padded_batch


def get_checkpoint_path(reload_path, best=True):
    import glob
    import numpy as np
    checkpoint_paths = glob.glob(reload_path + "checkpoint-*")    
    # always save best and last, so best will always have the lowest number
    checkpoint_step_nums = [int(x.split("-")[-1]) for x in checkpoint_paths]
    if best:
        best_checkpoint = np.argmin(checkpoint_step_nums)
        best_checkpoint_path = checkpoint_paths[best_checkpoint]
        return best_checkpoint_path
    else: # get last checkpoint instead
        last_checkpoint = np.argmax(checkpoint_step_nums)
        last_checkpoint_path = checkpoint_paths[last_checkpoint]
        return last_checkpoint_path


if __name__ == "__main__":
    args = parser.parse_args()
    args.task = "nmt"
    args.langs = args.langs.split(",")
    logger.info(args)
    datasets.logging.set_verbosity_error()

    char = True

    if args.debug:
        args.logging_steps = 1
        args.eval_steps = 10


        
    pretrained = 'google/byt5-small'
    tok = ByT5Tokenizer.from_pretrained(pretrained)
    tok_word = T5Tokenizer.from_pretrained('t5-small')
    if args.spm_model:
        tok_word = T5Tokenizer(args.spm_model, model_max_length=512)

    tok.model_input_names = ["input_ids", "input_word_ids", "input_word_lens", "attention_mask"] # for generation, need to specify extra inputs

    args.tok = tok
    args.tok.bos = args.tok.vocab_size - 1
    args.tok.eow = args.tok.vocab_size - 2
    args.tok.spm = args.tok.vocab_size - 3
    args.tok.bow = args.tok.vocab_size - 4
    args.tok.bos_buffer = args.bos_buffer
    args.tok.langs = args.langs

    args.word_tok = tok_word

    if args.dataset == "entr":
        assert "en" in args.langs and "tr" in args.langs
        dataset = load_dataset('dataloading/translation_entr.py', 'en-tr', cache_dir=args.cache_dir)
    elif args.dataset == "enfi":
        assert "en" in args.langs and "fi" in args.langs
        dataset = load_dataset('dataloading/translation_enfi.py', 'en-fi', cache_dir=args.cache_dir)
    elif args.dataset == "xhzu":
        assert "xh" in args.langs and "zu" in args.langs
        dataset = load_dataset('dataloading/translation_xhzu.py', 'xh-zu', cache_dir=args.cache_dir)
    elif args.dataset == "wmt14":
        assert "en" in args.langs and "de" in args.langs
        dataset = load_dataset('dataloading/wmt14_deen.py', '%s-%s' % tuple(args.langs), cache_dir=args.cache_dir)
    elif args.dataset == "flores":
        dataset = load_dataset('dataloading/translation_flores.py', '%s-%s' % tuple(args.langs), cache_dir=args.cache_dir)
    elif args.dataset == "biomed":
        assert "en" in args.langs and "de" in args.langs
        dataset = load_dataset('dataloading/translation_biomed.py', '%s-%s' % tuple(args.langs), cache_dir=args.cache_dir)
    else:
        dataset = load_dataset('dataloading/iwslt2017_dataset.py', 'iwslt2017-%s-%s' % tuple(args.langs), cache_dir=args.cache_dir)

    if args.data_lim: # limit amount of training data
        dataset['train'] = dataset['train'].filter(lambda example, idx: idx < args.data_lim, with_indices=True)

    if args.debug:
        dataset['train'] = dataset['train'].filter(lambda example, idx: idx < 1024, with_indices=True)
        dataset['validation'] = dataset['validation'].filter(lambda example, idx: idx < 1024, with_indices=True)
        dataset['test'] = dataset['test'].filter(lambda example, idx: idx < 1024, with_indices=True)

    # filter out sentence pairs where english side is longer than 256 chars (allows for larger batch sizes)
    primary_lang = 'en' if 'en' in args.langs else args.langs[0]
    dataset['train'] = dataset['train'].filter(lambda example: len(example['translation'][primary_lang]) < 256, num_proc=10)

    if args.filter_both_langs:
        secondary_lang = args.langs[1 - args.langs.index(primary_lang)]
        dataset['train'] = dataset['train'].filter(lambda example: len(example['translation'][secondary_lang]) < 256, num_proc=10)

    print("After filtering:", dataset['train'])

    if args.low_cache_size:
        tokenize = partial(tokenize_sdd_to_collate, char_tokenizer=tok, word_tokenizer=tok_word)
    else:
        tokenize = partial(tokenize_sdd, char_tokenizer=tok, word_tokenizer=tok_word)

    dataset = dataset.map(tokenize, batched=True, num_proc=10, load_from_cache_file=not args.recache_tok)
    print(dataset['train'][0])
 
    if args.clean_len > 0:
        clean_fn = lambda example: max(example['decoder_input_word_lens']) < args.clean_len
        dataset['train'] = dataset['train'].filter(clean_fn, num_proc=10)
        print("After cleaning:", dataset['train'])

    if args.set_max_word_len:
        max_word_len = args.set_max_word_len
    else:
        max_word_len = max([max(x['decoder_input_word_lens']) for x in dataset['train']])
    print("max word len:", max_word_len)

    args.save_path, reload_id = get_reload_path(args)
    args.reload_path = args.save_path if args.reload_id else None

    config = BertConfig(
        vocab_size=args.tok.vocab_size, 
        hidden_size=args.emb_dim, 
        num_hidden_layers=6 if not args.tiny else 1, 
        num_attention_heads=8, 
        intermediate_size=args.emb_dim*4,
        max_position_embeddings=1024,
        )

    if args.reload_path:
        args.checkpoint_path = get_checkpoint_path(args.reload_path, best=not args.resume)
        model = BertWithWordLee.from_pretrained(
            args.checkpoint_path, 
            config, 
            char_tok=args.tok if args.use_word_inputs else None,
            word_tok=tok_word if args.use_word_inputs else None,
            max_word_len=max_word_len+1)
    else:
        args.checkpoint_path = None
        model = BertWithWordLee(
            pretrained, 
            bert_config=config, 
            char_tok=args.tok if args.use_word_inputs else None,
            word_tok=tok_word if args.use_word_inputs else None,
            max_word_len=max_word_len+1, 
            )

    def get_num_params(model):
        total = 0
        for param in model.parameters():
            total += param.numel()
        return total

    logger.info("----- Model Parameters -----")
    logger.info("Total: %d" % get_num_params(model))
    # logger.info("Enc Embeds: %d" % get_num_params(model.bert.encoder.embeddings))
    # logger.info("Dec Embeds: %d" % get_num_params(model.bert.decoder.bert.embeddings))
    logger.info("----------------------------")

    model.config.bos_token_id = args.tok.bos
    model.config.eow_token_id = args.tok.eow
    model.config.bow_token_id = args.tok.bow

    experiment_name = "%s_%s" % (args.model_type, reload_id)

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
        warmup_steps=10000,
        eval_accumulation_steps=1,
        learning_rate=args.lr,
        label_smoothing_factor=0.1,
        load_best_model_at_end=True,
        disable_tqdm=not args.debug)


    early_stopping = EarlyStoppingCallback(early_stopping_patience=10)
    print_cb = LogFlushCallback(logger)
    cbs = [early_stopping, print_cb]

    compute_metrics = partial(compute_bleu, args=args)

    collator = padding_collate_fn if not args.low_cache_size else partial(padding_collate_fn_low_mem, char_tokenizer=tok)
    collator = collator if not args.batch_by_tokens \
        else partial(prep_batch_for_collating, collate_fn=collator)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tok,
        args=training_args, 
        train_dataset=dataset['train'], 
        eval_dataset=dataset['validation'],
        data_collator=collator,
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