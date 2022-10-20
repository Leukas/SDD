# lee_bert.py
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, EncoderDecoderModel
from transformers.models.bert.modeling_bert import BertModel, BertLMHeadModel
from models.lee import CharToPseudoWord, CharToWord
from models.charformer_bert import BertEncoderOnlyGBST, BertWithGBSTEncDec
from typing import Optional
import copy
from einops import rearrange
from transformers.generation_stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)


class CharEmbedder(nn.Module):
    def __init__(self, vocab_size, char_emb_dim, dim, ds_factor, decoder=False, max_pos_embs=1024):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, char_emb_dim)
        self.dropout = nn.Dropout(0.1)
        self.pos = nn.Parameter(    
            torch.randn(1, max_pos_embs, char_emb_dim))
            # torch.randn(1, 2048, char_emb_dim))
        self.lee = CharToPseudoWord(
            char_emb_dim, 
            intermediate_dim=dim,
            max_pool_window=ds_factor,
            is_decoder=decoder)

    def forward(self, x, mask):
        if mask is None:
            mask = torch.ones((x.size(0), x.size(1)), device=x.device)
        x = self.dropout(self.emb(x)) + self.pos[:, :x.size(1)]
        return self.lee(x, mask.float())

class CharEmbedderWord(nn.Module):
    def __init__(self, vocab_size, char_emb_dim, dim, decoder=False, max_pos_embs=2048):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, char_emb_dim)
        self.dropout = nn.Dropout(0.1)
        self.pos = nn.Parameter(    
            # torch.randn(1, 1024, char_emb_dim))
            torch.randn(1, max_pos_embs, char_emb_dim))
        self.lee = CharToWord(
            char_emb_dim, 
            intermediate_dim=dim,
            is_decoder=decoder)

    def forward(self, x, mask, word_lens):
        if mask is None:
            mask = torch.ones((x.size(0), x.size(1)), device=x.device)
        x = self.dropout(self.emb(x)) + self.pos[:, :x.size(1)]
        embed, mask = self.lee(x, mask.float(), word_lens)

        return embed, mask

class BertEncoderOnlyLee(BertEncoderOnlyGBST):
    def __init__(self, pretrained_model, bert_config=None, ds_factor=4, max_pos_embs=1024):
        super().__init__(pretrained_model, bert_config, ds_factor)

        self.gbst = CharEmbedder(self.vocab_size, 64, self.dim, self.ds_factor, False, max_pos_embs)

class BertEncoderOnlyLeeWord(nn.Module):
    def __init__(self, pretrained_model, bert_config=None, max_pos_embs=2048):
        super().__init__()
        if bert_config is not None:
            self.bert = BertForSequenceClassification(bert_config)
        else:
            raise NotImplementedError

        self.config = self.bert.config
        self.dim = self.config.hidden_size
        self.vocab_size = self.config.vocab_size


        self.lee = CharEmbedderWord(self.vocab_size, 64, self.dim, False, max_pos_embs=max_pos_embs)

    @classmethod
    def from_pretrained(cls, reload_folder, config=None):
        model = cls(None, config)
        state_dict_path = reload_folder + "/pytorch_model.bin"
        model.load_state_dict(torch.load(state_dict_path), strict=False)
        return model

    def embed(self, input_ids, input_word_lens, attention_mask=None):
        embed, _ = self.lee(
            input_ids, 
            attention_mask.bool() if attention_mask is not None else None,
            input_word_lens)
    
        return embed

    def forward(self, input_ids, input_word_lens, attention_mask=None, labels=None):
        embed, mask = self.lee(
            input_ids, 
            attention_mask.bool() if attention_mask is not None else None,
            input_word_lens)

        out = self.bert(
            inputs_embeds=embed, 
            attention_mask=mask,
            labels=labels)

        return out

class BertWithLeeFull(BertWithGBSTEncDec):
    def __init__(self, pretrained_model, bert_config=None, causal=True, ds_factor=4, max_pos_embs=1024):
        super().__init__(pretrained_model, bert_config, causal, ds_factor)

        self.enc_gbst = CharEmbedder(self.vocab_size, 64, self.dim, self.ds_factor, False, max_pos_embs)
        self.dec_gbst = CharEmbedder(self.vocab_size, 64, self.dim, self.ds_factor, True, max_pos_embs)

    @classmethod
    def from_pretrained(cls, reload_folder, config=None, causal=True, ds_factor=4, max_pos_embs=1024):
        model = cls(None, config, causal, ds_factor, max_pos_embs)
        state_dict_path = reload_folder + "/pytorch_model.bin"
        model.load_state_dict(torch.load(state_dict_path))
        return model


class BertWithWordLee(nn.Module):
    def __init__(self, pretrained_model, bert_config=None, max_word_len=8, char_tok=None, word_tok=None, max_pos_embs=2048):
        super().__init__()
        if bert_config is not None:
            enc = BertModel(bert_config)
            bert_config.is_decoder = True
            bert_config.add_cross_attention = True
            dec = BertLMHeadModel(bert_config)
            self.bert = EncoderDecoderModel(encoder=enc, decoder=dec)
        else:
            raise NotImplementedError

        self.config = self.bert.config.encoder
        self.dim = self.config.hidden_size
        self.vocab_size = self.config.vocab_size

        self.max_word_len = max_word_len
        self.enc_lee = CharEmbedderWord(self.vocab_size, 64, self.dim, False, max_pos_embs=max_pos_embs)
        self.dec_lee = CharEmbedderWord(self.vocab_size, 64, self.dim, True, max_pos_embs=max_pos_embs)

        self.decoding_proj_layer = nn.Linear(self.dim, 64*self.max_word_len)
        self.decoding_char_embs = nn.Embedding(self.vocab_size, 64)

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True)

        self.pred_layer = nn.Linear(128, self.vocab_size)

        if char_tok is not None:
            self.char_tok = char_tok
            self.word_tok = word_tok

    @classmethod
    def from_pretrained(cls, reload_folder, config=None, max_word_len=8, char_tok=None, word_tok=None):
        model = cls(None, config, max_word_len, char_tok=char_tok, word_tok=word_tok)
        state_dict_path = reload_folder + "/pytorch_model.bin"
        model.load_state_dict(torch.load(state_dict_path))
        return model

    def forward(self, 
        input_ids, 
        input_word_lens,
        input_word_ids=None,
        attention_mask=None, 
        decoder_input_ids=None, 
        decoder_input_word_lens=None, 
        decoder_input_word_ids=None, 
        decoder_attention_mask=None, 
        labels=None,
        input_bounds=None,
        lstm_input_ids=None,
        lstm_word_lens=None,

        ):

        embed, mask = self.enc_lee(
            input_ids, 
            attention_mask.bool() if attention_mask is not None else None,
            input_word_lens)
        
        dec_embed, dec_mask = self.dec_lee(
            decoder_input_ids, 
            decoder_attention_mask.bool() if decoder_attention_mask is not None else None,
            decoder_input_word_lens, 
            )

        out = self.bert(
            inputs_embeds=embed, 
            attention_mask=mask,
            decoder_inputs_embeds=dec_embed,
            decoder_attention_mask=dec_mask,
            output_hidden_states=True,
            return_dict=True)

        batch_size = input_ids.size(0)
        dec_char_embs = self.decoding_char_embs(lstm_input_ids) # b p d
        hidden_proj = self.decoding_proj_layer(out.decoder_hidden_states[-1])
        repeated_out = rearrange(hidden_proj, 'b n (m d) -> b n m d', m=self.max_word_len)
        selection_mask = torch.stack([torch.arange(self.max_word_len, device=lstm_word_lens.device) < lstm_word_lens[i][:, None] \
            for i in range(batch_size)]) # b n m
        masked_repeated = [repeated_out[i][selection_mask[i]] for i in range(batch_size)] # b * [k d]
        padded_masked_repeated = torch.zeros_like(dec_char_embs)
        for i in range(batch_size):
            padded_masked_repeated[i, :len(masked_repeated[i])] = masked_repeated[i][:dec_char_embs.size(1)]

        concatenated_lstm_input = torch.cat([dec_char_embs, padded_masked_repeated], dim=-1)
        lstm_out, _ = self.lstm(concatenated_lstm_input)
        lm_logits = self.pred_layer(lstm_out)

        return (lm_logits,)

    def encode(self, input_ids, input_word_lens, attention_mask, input_word_ids):
        embed, mask = self.enc_lee(
            input_ids, 
            attention_mask.bool() if attention_mask is not None else None,
            input_word_lens 
            )
        out = self.bert.encoder(
            inputs_embeds=embed, 
            attention_mask=mask,
            return_dict=True)
        return out.last_hidden_state, mask

    def decode_lstm_generate(self, 
        decoder_input_ids, 
        decoder_input_word_lens,
        decoder_input_word_ids,
        decoder_attention_mask, 
        encoder_hidden_states, 
        encoder_attention_mask,
        last_lstm_input_id,
        lstm_hidden_state,
        past_key_values
        ):
        batch_size = decoder_input_ids.size(0)

        dec_embed, dec_mask = self.dec_lee(
            decoder_input_ids,
            decoder_attention_mask.bool() if decoder_attention_mask is not None else None,
            decoder_input_word_lens
        )

        assert encoder_hidden_states is not None
        out = self.bert.decoder(
            inputs_embeds=dec_embed[:, -1:], 
            attention_mask=dec_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
            use_cache=True,
            past_key_values=past_key_values,
            return_dict=True)

        last_hidden_token = out.hidden_states[-1][:, -1].unsqueeze(1)

        dec_char_embs = self.decoding_char_embs(last_lstm_input_id[:, -1].unsqueeze(1))
        hidden_proj = self.decoding_proj_layer(last_hidden_token)
        repeated_out = rearrange(hidden_proj, 'b n (m d) -> b (n m) d', m=self.max_word_len)

        concatenated_lstm_input = torch.cat([dec_char_embs, repeated_out[:, 0].unsqueeze(1)], dim=-1)

        finished = torch.zeros(batch_size, device=decoder_input_ids.device).bool()
        block_lens = torch.zeros(batch_size, device=decoder_input_ids.device).long()
        all_preds = []
        lstm_0 = torch.zeros(1, batch_size, self.lstm.hidden_size, device=decoder_input_ids.device) \
            if lstm_hidden_state is None else lstm_hidden_state[0]
        lstm_1 = torch.zeros(1, batch_size, self.lstm.hidden_size, device=decoder_input_ids.device) \
            if lstm_hidden_state is None else lstm_hidden_state[1]
        for i in range(self.max_word_len):
            lstm_out, lstm_hid = self.lstm(concatenated_lstm_input, (lstm_0, lstm_1))
            lm_logits = self.pred_layer(lstm_out)
            preds = lm_logits.argmax(dim=-1)

            lstm_0[0][~finished] = lstm_hid[0][0][~finished] # shape is (1, bs, dim)
            lstm_1[0][~finished] = lstm_hid[1][0][~finished]

            preds[finished] = 0
            all_preds.append(preds)

            eows = (preds == self.config.eow_token_id).squeeze()
            finished[eows] = 1
            block_lens[~finished] += 1

            if finished.min() == 1:
                break


            if i < self.max_word_len-1: # updates for next lstm iteration
                pred_embs = self.decoding_char_embs(preds)
                concatenated_lstm_input = torch.cat([pred_embs, repeated_out[:, i+1].unsqueeze(1)], dim=-1)

        return torch.cat(all_preds, dim=1), block_lens, (lstm_0, lstm_1), out.past_key_values

    def generate(self, input_ids, input_word_lens, attention_mask=None, input_word_ids=None, **kwargs):
        enc, enc_mask = self.encode(input_ids, input_word_lens, attention_mask, input_word_ids)
        
        bos_token_id = \
            kwargs["decoder_start_token_id"] if "decoder_start_token_id" in kwargs \
            else self.config.bos_token_id # NOTE: check this line if it works with nmt.py, otherwise add decoder_start_token_id to kwargs
        
        assert self.config.bos_token_id == bos_token_id, str(self.config.bos_token_id) + ", " + str(bos_token_id) # To remove from non-nmt.py runs

        decoder_start = torch.full((input_ids.size(0), 1), bos_token_id).type_as(input_ids)
        decoder_start_word = torch.full((input_ids.size(0), 1), bos_token_id).type_as(input_ids)
        
        return self.greedy_search_lstm(
            input_ids=decoder_start, 
            input_word_ids=decoder_start_word, 
            encoder_hidden_states=enc,
            encoder_attention_mask=enc_mask,
            **kwargs)
        
    def greedy_search_lstm(
        self,
        input_ids: torch.FloatTensor,
        input_word_ids: torch.FloatTensor,
        logits_processor = None,
        stopping_criteria = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> torch.LongTensor:
        """ Generate function to work with the 2-step LSTM decoding proposed by Libovicky et al. 2021, code adapted from HuggingFace """
        # init values
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        # keep track of which sequences are already finished
        # unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        batch_size = input_ids.shape[0]
        unfinished_sequences = torch.ones(batch_size).to(input_ids.device)
        dec_word_lens = torch.ones(batch_size, 1).to(input_ids.device)
        dec_seq_lens = torch.ones(batch_size).long().to(input_ids.device)
        # dec_seq_lens_with_eow = torch.ones(batch_size).long().to(input_ids.device)
        dec_attn_mask = torch.ones(batch_size, 1).bool().to(input_ids.device)
        # cur_len = input_ids.shape[-1]
        full_sequence = torch.zeros((batch_size, max_length*self.max_word_len), device=input_ids.device).long()   
        full_sequence[:, :input_ids.size(1)] = input_ids

        lstm_input_ids = copy.deepcopy(input_ids)
        past_key_values = None
        lstm_hidden = None
        if hasattr(self, 'char_tok'):
            spm_token_str = self.char_tok.decode([self.char_tok.spm])
        while True:
            # forward pass to get next token
            with torch.no_grad():
                next_tokens, block_lens, lstm_hidden, past_key_values = self.decode_lstm_generate(
                    decoder_input_ids=input_ids,
                    decoder_input_word_lens=dec_word_lens,
                    decoder_input_word_ids=input_word_ids,
                    decoder_attention_mask=dec_attn_mask,
                    encoder_hidden_states=model_kwargs['encoder_hidden_states'],
                    encoder_attention_mask=model_kwargs['encoder_attention_mask'],
                    last_lstm_input_id=lstm_input_ids,
                    lstm_hidden_state=lstm_hidden,
                    past_key_values=past_key_values,
                )

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences.unsqueeze(1) + (pad_token_id * (1 - unfinished_sequences)).unsqueeze(1)

            # update generated ids, model inputs, and length for next step
            new_dec_seq_lens = dec_seq_lens + block_lens

            for b in range(batch_size):
                full_sequence[b, dec_seq_lens[b]:new_dec_seq_lens[b]] = next_tokens[b][:block_lens[b]]

                # NOTE: Apart from the first word, the last char should be an eow anyway
                lstm_input_ids[b, -1] = self.config.eow_token_id

            # word combining
            next_words = torch.zeros(batch_size, 1).type_as(input_word_ids)

            if hasattr(self, 'char_tok'):
                for b in range(batch_size):
                    b_str = self.char_tok.decode(next_tokens[b][:block_lens[b]])
                    b_str = b_str.replace(spm_token_str,"▁")
                    b_str_prev = self.word_tok.decode(input_word_ids[b])
                    b_full_str = b_str_prev + b_str # full string sentence so far
                    next_word_idx = len(self.word_tok.tokenize(b_str_prev)) # idx of the next word
                    next_word = self.word_tok.encode(b_full_str)[next_word_idx] # get first token

                    next_words[b] = next_word

            input_word_ids = torch.cat([input_word_ids, next_words], dim=1)
            input_ids = full_sequence[:,:new_dec_seq_lens.max()]

            # update dec idxs
            dec_seq_lens = new_dec_seq_lens
            dec_attn_mask = torch.arange(new_dec_seq_lens.max(), device=new_dec_seq_lens.device) < new_dec_seq_lens[:, None]
            dec_word_lens = torch.cat([dec_word_lens, block_lens.unsqueeze(1)], dim=1)

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished = (next_tokens != eos_token_id).min(dim=-1)[0]
                unfinished_sequences = unfinished_sequences.mul(unfinished.long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break

        return input_ids