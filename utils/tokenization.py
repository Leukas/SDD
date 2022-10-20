# tokenization.py
import torch
import numpy as np

def align_tok_sents(char_tok, word_tok, sentence):
    tokw = word_tok.tokenize(sentence)
    tokw_str = " ".join(tokw) # string with bpe applied
    tokc = char_tok.tokenize(tokw_str) # string with bpe applied, multi-byte chars split up
    tokc_str = "".join(tokc) # concat back

    tokc_str_1bpe = tokc_str.replace('â\x96\x81', '.') # replace the 3-byte with 1 byte

    tokc_enc = char_tok.encode(tokw_str.replace(" ","").replace("▁"," "))

    word_lens = [len(word) for word in tokc_str_1bpe.split(" ")] + [1]

    return tokc_enc, word_lens

def tokenize_sdd(examples, char_tokenizer, word_tokenizer, include_bow=False):
    encoded = {
        'input_ids': [], 
        'input_word_lens': [], 
        'attention_mask': [], 
        'decoder_input_ids': [], 
        'decoder_input_word_lens': [], 
        'decoder_attention_mask': [], 
        'lstm_input_ids': [],
        'lstm_word_lens': [],
        'labels': []}

    for example in examples['translation']:
        src = example[char_tokenizer.langs[0]]
        tgt = example[char_tokenizer.langs[1]]


        src_tok, src_lens = align_tok_sents(char_tokenizer, word_tokenizer, src)
        # print("src", src_tok, src_lens)
        encoded['input_ids'] += [src_tok]
        encoded['input_word_lens'] += [src_lens]
        
        tgt_tok, tgt_lens = align_tok_sents(char_tokenizer, word_tokenizer, tgt)
        encoded['decoder_input_ids'] += [[char_tokenizer.bos] + tgt_tok[:-1]]

        dec_word_lens = tgt_lens[:-1] # remove EOS for now

        buffered_labels = tgt_tok[:]
        lstm_word_lens = []
        cum_idxs = np.cumsum(dec_word_lens)
        for i in range(len(dec_word_lens)):
            idx = cum_idxs[i]
            bidx = 0 if i == 0 else cum_idxs[i-1]
            if include_bow:
                buffered_labels.insert(bidx+2*i, char_tokenizer.bow)
                buffered_labels.insert(idx+2*i+1, char_tokenizer.eow)
            else:
                buffered_labels.insert(idx+i, char_tokenizer.eow)
            lstm_word_lens.append(dec_word_lens[i]+1)

        buffered_tgt_tok = buffered_labels[:]
        encoded['lstm_input_ids'] += [[char_tokenizer.bos] + buffered_tgt_tok[:-1]]
        encoded['labels'] += [buffered_labels]
        encoded['lstm_word_lens'] += [lstm_word_lens + [1]] # add EOS

        encoded['decoder_input_word_lens'] += [[1] + dec_word_lens] # add BOS

        attention_mask = torch.ones(len(src_tok), dtype=torch.long).tolist()
        encoded['attention_mask'] += [attention_mask]

        dec_attention_mask = torch.ones(len(tgt_tok), dtype=torch.long).tolist()
        encoded['decoder_attention_mask'] += [dec_attention_mask]
    return encoded

def tokenize_sdd_to_collate(examples, char_tokenizer, word_tokenizer):
    encoded = {
        'input_ids': [], 
        'input_word_lens': [], 
        'decoder_input_ids': [], 
        'decoder_input_word_lens': [], 
        }

    for example in examples['translation']:
        src = example[char_tokenizer.langs[0]]
        tgt = example[char_tokenizer.langs[1]]


        src_tok, src_lens = align_tok_sents(char_tokenizer, word_tokenizer, src)
        # print("src", src_tok, src_lens)
        encoded['input_ids'] += [src_tok]
        encoded['input_word_lens'] += [src_lens]
        
        tgt_tok, tgt_lens = align_tok_sents(char_tokenizer, word_tokenizer, tgt)
        encoded['decoder_input_ids'] += [[char_tokenizer.bos] + tgt_tok] # NOTE: remove last token when collating
        dec_word_lens = tgt_lens[:-1] # remove EOS for now
        encoded['decoder_input_word_lens'] += [[1] + dec_word_lens] # add BOS

    return encoded

def padding_collate_fn_low_mem(batch, max_len=1024, char_tokenizer=None):
    """ 
        Pads each list with zeros and concatenates by key.
        Input: List[{key: List[], ...}]
        Output: {key: LongTensor(), ...}
    """
    # pre-collating stuff that would normally be in tokenization fn
    for b in range(len(batch)):
        src_tok = batch[b]['input_ids']
        tgt_tok = batch[b]['decoder_input_ids'][1:]
        dec_word_lens = batch[b]['decoder_input_word_lens'][1:]

        buffered_labels = tgt_tok[:]
        lstm_word_lens = []
        cum_idxs = np.cumsum(dec_word_lens)
        for i in range(len(dec_word_lens)):
            idx = cum_idxs[i] -1
            buffered_labels.insert(idx+i+1, char_tokenizer.eow)
            lstm_word_lens.append(dec_word_lens[i]+1)

        buffered_tgt_tok = buffered_labels[:]
        batch[b]['lstm_input_ids'] = [char_tokenizer.bos] + buffered_tgt_tok[:-1]
        batch[b]['labels'] = buffered_labels
        batch[b]['lstm_word_lens'] = lstm_word_lens + [1] # add EOS


        attention_mask = torch.ones(len(src_tok), dtype=torch.long).tolist()
        batch[b]['attention_mask'] = attention_mask

        dec_attention_mask = torch.ones(len(tgt_tok), dtype=torch.long).tolist()
        batch[b]['decoder_attention_mask'] = dec_attention_mask

        batch[b]['decoder_input_ids'] = batch[b]['decoder_input_ids'][:-1] # remove last token

    # actual collating
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

def tokenize_encoder_only_sdd(examples, char_tokenizer, word_tokenizer, src_field, tgt_field):
    encoded = {
        'input_ids': [], 
        'input_word_lens': [], 
        'attention_mask': [], 
        'labels': []}

    for i in range(len(examples[src_field])):
        src = examples[src_field][i]
        tgt = examples[tgt_field][i]

        src_enc, src_word_lens = align_tok_sents(char_tokenizer, word_tokenizer, src)
        tgt_enc, tgt_word_lens = align_tok_sents(char_tokenizer, word_tokenizer, tgt)

        total_lens = [1] + src_word_lens + [1] + tgt_word_lens # add cls and sep token in

        inp = [char_tokenizer.cls] + src_enc + [char_tokenizer.sep] + tgt_enc 

        encoded['input_ids'] += [inp]
        encoded['input_word_lens'] += [total_lens]
        encoded['labels'] += [examples['label'][i]]

        attention_mask = torch.ones(len(inp), dtype=torch.long).tolist()
        encoded['attention_mask'] += [attention_mask]

    return encoded

def tokenize_nli_sdd(examples, char_tokenizer, word_tokenizer):
    return tokenize_encoder_only_sdd(examples, char_tokenizer, word_tokenizer, src_field='premise', tgt_field='hypothesis')

def tokenize_rc_sdd(examples, char_tokenizer, word_tokenizer):
    return tokenize_encoder_only_sdd(examples, char_tokenizer, word_tokenizer, src_field='review_headline', tgt_field='review_body')
