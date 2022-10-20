# utils.py
# Contains random utility functions that are useful for multiple tasks
import os
import glob
from copy import deepcopy
import torch
import numpy as np

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
        if key == "labels":
            padded_batch[key] -= 100
    
    for i, sample in enumerate(batch):
        for key in padded_batch:
            key_len = min(max_len, len(sample[key]))
            padded_batch[key][i, :key_len] = torch.LongTensor(sample[key][:key_len])

    return padded_batch

def tokenize(examples, tokenizer, buffer_ds_factor=0):
    encoded = {
        'input_ids': [], 
        'attention_mask': [], 
        'decoder_input_ids': [], 
        'decoder_attention_mask': [], 
        'labels': []}

    for example in examples['translation']:
        src = example[tokenizer.langs[0]]
        tgt = example[tokenizer.langs[1]]

        if buffer_ds_factor:
            src = buffer_by_ds_factor(src, buffer_ds_factor)
            tgt = buffer_by_ds_factor(tgt, buffer_ds_factor)

        src_tok = tokenizer.encode(src)
        encoded['input_ids'] += [src_tok]
        
        tgt_tok = tokenizer.encode(tgt)
        encoded['decoder_input_ids'] += [[tokenizer.bos]*tokenizer.bos_buffer + tgt_tok[:-tokenizer.bos_buffer]]
        encoded['labels'] += [tgt_tok]

        attention_mask = torch.ones(len(src_tok), dtype=torch.long).tolist()
        encoded['attention_mask'] += [attention_mask]

        dec_attention_mask = torch.ones(len(tgt_tok), dtype=torch.long).tolist()
        encoded['decoder_attention_mask'] += [dec_attention_mask]
    return encoded

def buffer_by_ds_factor(sent, ds_factor):
    byte_sent = sent.encode('utf8').split(b" ")
    for i in range(len(byte_sent)):
        toklen = len(byte_sent[i])
        buffer_str = b" " * (ds_factor - toklen % ds_factor) 
        byte_sent[i] = byte_sent[i] + buffer_str

    new_sent = b"".join(byte_sent).decode('utf8')
    return new_sent

def get_reload_path(args):
    model_type = args.diff_pretrained_type if hasattr(args, 'diff_pretrained_type') and args.diff_pretrained_type else args.model_type
    reload_id = args.reload_id if args.reload_id else os.environ.get('SLURM_JOB_ID')
    if reload_id is None: # no slurm, no reload
        import random
        chars = '0123456789'
        while True:
            reload_id = ''.join(random.choice(chars) for _ in range(5))
            if not os.path.isdir("dumped/%s/%s/%s/" % (args.task, model_type, reload_id)):
                break
        print("Reload id assigned is:", reload_id)

    reload_path = "dumped/%s/%s/%s/" % (args.task, model_type, reload_id)
    return reload_path, reload_id

def get_checkpoint_path(reload_path, best=True):
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

def load_model(pretrained, args):
    from transformers import EncoderDecoderModel
    from transformers.models.bert.configuration_bert import BertConfig
    from transformers.models.bert.modeling_bert import BertModel, BertLMHeadModel

    config = BertConfig(
        vocab_size=args.tok.vocab_size, 
        hidden_size=args.emb_dim, 
        num_hidden_layers=6 if not args.tiny else 3, 
        num_attention_heads=8, 
        intermediate_size=args.emb_dim*4,
        max_position_embeddings=args.max_pos_embs,
        ) # Transformer Base

    if args.reload_path:
        pretrained = get_checkpoint_path(args.reload_path, args.reload_best)
        
    if args.model_type == "gbst":
        from models.charformer_bert import BertWithGBSTEncDec
        if args.reload_path:
            model = BertWithGBSTEncDec.from_pretrained(pretrained, config, causal=True, ds_factor=args.ds_factor)
        else:
            model = BertWithGBSTEncDec(pretrained, bert_config=config, causal=True, ds_factor=args.ds_factor)
        return model

    if args.model_type == "fixed" or args.model_type == "buffered_fixed":
        from models.lee_bert import BertWithLeeFull
        if args.reload_path:
            model = BertWithLeeFull.from_pretrained(pretrained, config, causal=True, ds_factor=args.ds_factor, max_pos_embs=args.max_pos_embs)
        else:
            model = BertWithLeeFull(pretrained, bert_config=config, causal=True, ds_factor=args.ds_factor, max_pos_embs=args.max_pos_embs)
        return model

    if args.model_type == "twostep" or args.model_type == "twostep_word":
        from models.twostep import TwoStep
        if args.reload_path:
            model = TwoStep.from_pretrained(pretrained, config)
        else:
            model = TwoStep(pretrained, bert_config=config)
        return model
    
    if "word" in args.model_type or args.model_type == "char":
        if args.reload_path:
            model = EncoderDecoderModel.from_pretrained(pretrained)
        else:
            enc_config = deepcopy(config)
            enc_model = BertModel(enc_config)
            
            dec_config = deepcopy(config)
            dec_config.is_decoder = True
            dec_config.add_cross_attention = True
            dec_model = BertLMHeadModel(dec_config)

            model = EncoderDecoderModel(encoder=enc_model, decoder=dec_model)
        return model
    
    assert False, "%s is not a valid model type." % args.model_type



def load_model_encoder(pretrained, args):
    from transformers.models.bert.configuration_bert import BertConfig

    config = BertConfig(
        vocab_size=args.tok.vocab_size, 
        hidden_size=args.emb_dim, 
        num_hidden_layers=6 if not args.tiny else 3, 
        num_attention_heads=8, 
        intermediate_size=args.emb_dim*4,
        max_position_embeddings=1024,
        num_labels=args.num_labels if 'num_labels' in args else 3,
        ) # Transformer Base

    if args.reload_path:
        pretrained = get_checkpoint_path(args.reload_path, args.reload_best)

    if args.model_type == "gbst":
        from models.charformer_bert import BertEncoderOnlyGBST
        if args.reload_path:
            model = BertEncoderOnlyGBST.from_pretrained(pretrained, config, ds_factor=args.ds_factor)
        else:
            model = BertEncoderOnlyGBST(pretrained, bert_config=config, ds_factor=args.ds_factor)
        return model
        
    if args.model_type == "fixed" or args.model_type == "buffered_fixed":
        from models.lee_bert import BertEncoderOnlyLee
        if args.reload_path:
            model = BertEncoderOnlyLee.from_pretrained(pretrained, config, ds_factor=args.ds_factor)
        else:
            model = BertEncoderOnlyLee(pretrained, bert_config=config, ds_factor=args.ds_factor)
        return model

    if args.model_type == "sdd":
        from models.lee_bert import BertEncoderOnlyLeeWord
        if args.reload_path:
            model = BertEncoderOnlyLeeWord.from_pretrained(pretrained, config)
        else:
            model = BertEncoderOnlyLeeWord(pretrained, bert_config=config)
        return model

    if "word" in args.model_type or args.model_type == "char":
        from transformers.models.bert.modeling_bert import BertForSequenceClassification
        if args.reload_path:
            model = BertForSequenceClassification.from_pretrained(pretrained)
        else:
            model = BertForSequenceClassification(config)
        return model
    

    assert False, "%s is not a valid model type." % args.model_type
