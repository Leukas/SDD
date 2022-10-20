import warnings
import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional
from transformers import EncoderDecoderModel
from transformers.models.bert.modeling_bert import BertModel, BertLMHeadModel, BertForSequenceClassification
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.generation_stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation_logits_process import LogitsProcessorList
from models.gbst import GBST

class BertEncoderOnlyGBST(nn.Module):
    def __init__(self, pretrained_model, bert_config=None, ds_factor=4):
        super().__init__()
        if bert_config is not None:
            self.bert = BertForSequenceClassification(bert_config)
        else:
            raise NotImplementedError

        self.config = self.bert.config
        self.dim = self.config.hidden_size
        self.vocab_size = self.config.vocab_size

        self.ds_factor = ds_factor
        self.gbst = GBST(
            num_tokens=self.vocab_size, 
            dim=self.dim, 
            downsample_factor=self.ds_factor,
            max_block_size=self.ds_factor, 
            score_consensus_attn=True, 
            causal=False)

    @classmethod
    def from_pretrained(cls, reload_folder, config=None, ds_factor=4):
        model = cls(None, config, ds_factor)
        state_dict_path = reload_folder + "/pytorch_model.bin"
        model.load_state_dict(torch.load(state_dict_path))
        return model

    def forward(self, input_ids, attention_mask=None, labels=None):

        # print("ids", input_ids.shape)
        # print("atm", attention_mask.shape)
        embed, mask = self.gbst(input_ids, mask=attention_mask.bool() if attention_mask is not None else None)
        out = self.bert(
            inputs_embeds=embed, 
            attention_mask=mask,
            labels=labels)

        return out


class BertWithGBSTEncoder(nn.Module):
    def __init__(self, pretrained_model, bert_config=None, ds_factor=4):
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

        self.ds_factor = ds_factor
        self.gbst = GBST(
            num_tokens=self.vocab_size, 
            dim=self.dim, 
            downsample_factor=self.ds_factor,
            max_block_size=self.ds_factor, 
            score_consensus_attn=True, 
            causal=False)

    @classmethod
    def from_pretrained(cls, reload_folder, config=None):
        model = cls(None, config)
        state_dict_path = reload_folder + "/pytorch_model.bin"
        model.load_state_dict(torch.load(state_dict_path))
        return model

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None):
        embed, mask = self.gbst(input_ids, mask=attention_mask.bool() if attention_mask is not None else None)
        out = self.bert(
            inputs_embeds=embed, 
            attention_mask=mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels)

        return out

    def generate(self, input_ids, attention_mask=None, **kwargs):
        embed, mask = self.gbst(input_ids, mask=attention_mask.bool() if attention_mask is not None else None)
        decoder_start = torch.full((input_ids.size(0), 1), self.config.bos_token_id).type_as(input_ids)
        out = self.bert.generate(
            inputs_embeds=embed, 
            attention_mask=mask,
            decoder_input_ids=decoder_start,
            **kwargs)

        return out


class BertWithGBSTEncDec(nn.Module):
    def __init__(self, pretrained_model, bert_config=None, causal=True, ds_factor=4):
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

        self.ds_factor = ds_factor
        self.enc_gbst = GBST(
            num_tokens=self.vocab_size, 
            dim=self.dim, 
            downsample_factor=self.ds_factor,
            max_block_size=4, 
            score_consensus_attn=True, 
            causal=False
            )
            
        
        self.dec_gbst = GBST(
            num_tokens=self.vocab_size, 
            dim=self.dim, 
            downsample_factor=self.ds_factor,
            max_block_size=self.ds_factor, 
            score_consensus_attn=True, 
            causal=causal,
            no_conv=True
            )

        out_dim = 64

        self.decoding_proj_layer = nn.Linear(self.dim, out_dim*self.ds_factor)
        self.decoding_char_embs = nn.Embedding(self.vocab_size, out_dim)

        self.lstm = nn.LSTM(
            input_size=2*out_dim,
            hidden_size=2*out_dim,
            num_layers=1,
            batch_first=True)

        self.pred_layer = nn.Linear(2*out_dim, self.vocab_size)

    @classmethod
    def from_pretrained(cls, reload_folder, config=None, causal=True, ds_factor=4):
        model = cls(None, config, causal, ds_factor)
        state_dict_path = reload_folder + "/pytorch_model.bin"
        model.load_state_dict(torch.load(state_dict_path))
        return model

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        embed, mask = self.enc_gbst(input_ids, mask=attention_mask.bool() if attention_mask is not None else None)
        bos_buffer = decoder_input_ids[:, 0].unsqueeze(1).repeat(1, self.ds_factor-1)
        buffered_dec_ids = torch.cat([bos_buffer, decoder_input_ids], dim=1)
        if decoder_attention_mask is not None:
            mask_buffer = torch.ones_like(decoder_attention_mask)[:, :self.ds_factor-1]
            buffered_dec_mask = torch.cat([mask_buffer, decoder_attention_mask], dim=1)
        
        dec_embed, dec_mask = self.dec_gbst(buffered_dec_ids, mask=buffered_dec_mask.bool() if decoder_attention_mask is not None else None)
        out = self.bert(
            inputs_embeds=embed, 
            attention_mask=mask,
            decoder_inputs_embeds=dec_embed,
            decoder_attention_mask=dec_mask,
            output_hidden_states=True,
            return_dict=True)

        dec_char_embs = self.decoding_char_embs(decoder_input_ids)
        hidden_proj = self.decoding_proj_layer(out.decoder_hidden_states[-1])
        repeated_out = rearrange(hidden_proj, 'b n (m d) -> b (n m) d', m=self.ds_factor)

        concatenated_lstm_input = torch.cat([dec_char_embs, repeated_out[:, :dec_char_embs.size(1)]], dim=-1)
        lstm_out, _ = self.lstm(concatenated_lstm_input)
        lm_logits = self.pred_layer(lstm_out)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            return Seq2SeqLMOutput(
                loss=loss,
                logits=lm_logits,
                past_key_values=out.past_key_values,
                decoder_hidden_states=out.decoder_hidden_states,
                decoder_attentions=out.decoder_attentions,
                cross_attentions=out.cross_attentions,
                encoder_last_hidden_state=out.encoder_last_hidden_state,
                encoder_hidden_states=out.encoder_hidden_states,
                encoder_attentions=out.encoder_attentions,
                )

        return (lm_logits,)

    def encode(self, input_ids, attention_mask):
        embed, mask = self.enc_gbst(input_ids, mask=attention_mask.bool() if attention_mask is not None else None)
        out = self.bert.encoder(
            inputs_embeds=embed, 
            attention_mask=mask,
            return_dict=True)
        return out.last_hidden_state, mask

    def decode_lstm_generate(self, decoder_input_ids, decoder_attention_mask, encoder_hidden_states, encoder_attention_mask, lstm_hidden_state, past_key_values):
        assert encoder_hidden_states is not None
        dec_embed, dec_mask = self.dec_gbst(decoder_input_ids, mask=decoder_attention_mask.bool() if decoder_attention_mask is not None else None)
        
        dec_embed = dec_embed[:, -1:] # NOTE: This is done here because Lee's method uses previous 
        # blocks. Could be done before calling dec_gbst to save computation time.
        
        out = self.bert.decoder(
            inputs_embeds=dec_embed, 
            attention_mask=dec_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
            use_cache=True,
            past_key_values=past_key_values,
            return_dict=True)

        last_hidden_token = out.hidden_states[-1][:, -1].unsqueeze(1)

        dec_char_embs = self.decoding_char_embs(decoder_input_ids[:, -1].unsqueeze(1))
        hidden_proj = self.decoding_proj_layer(last_hidden_token)
        repeated_out = rearrange(hidden_proj, 'b n (m d) -> b (n m) d', m=self.ds_factor)

        concatenated_lstm_input = torch.cat([dec_char_embs, repeated_out[:, 0].unsqueeze(1)], dim=-1)
        
        all_preds = []
        for i in range(self.ds_factor):
            lstm_out, lstm_hidden_state = self.lstm(concatenated_lstm_input, lstm_hidden_state)
            lm_logits = self.pred_layer(lstm_out)
            preds = lm_logits.argmax(dim=-1)
            all_preds.append(preds)

            if i < self.ds_factor-1: # updates for next lstm iteration
                pred_embs = self.decoding_char_embs(preds)
                concatenated_lstm_input = torch.cat([pred_embs, repeated_out[:, i+1].unsqueeze(1)], dim=-1)

        return torch.cat(all_preds, dim=1), lstm_hidden_state, out.past_key_values

    def generate(self, input_ids, attention_mask=None, **kwargs):
        enc, enc_mask = self.encode(input_ids, attention_mask)
        
        bos_token_id = \
            kwargs["decoder_start_token_id"] if "decoder_start_token_id" in kwargs \
            else self.config.bos_token_id # NOTE: check this line if it works with nmt.py, otherwise add decoder_start_token_id to kwargs
        
        assert self.config.bos_token_id == bos_token_id, str(self.config.bos_token_id) + ", " + str(bos_token_id) # To remove from non-nmt.py runs

        decoder_start = torch.full((input_ids.size(0), self.ds_factor), bos_token_id).type_as(input_ids)
                
        return self.greedy_search_lstm(
            input_ids=decoder_start, 
            encoder_hidden_states=enc,
            encoder_attention_mask=enc_mask,
            **kwargs)
        
    def greedy_search_lstm(
        self,
        input_ids: torch.FloatTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
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
        unfinished_sequences = torch.ones(input_ids.shape[0]).to(input_ids.device)
        cur_len = input_ids.shape[-1]

        past_key_values = None
        lstm_hidden = None
        while True:
            # forward pass to get next token
            with torch.no_grad():
                next_tokens, lstm_hidden, past_key_values = self.decode_lstm_generate(
                    decoder_input_ids=input_ids,
                    decoder_attention_mask=None,
                    encoder_hidden_states=model_kwargs['encoder_hidden_states'],
                    encoder_attention_mask=model_kwargs['encoder_attention_mask'],
                    lstm_hidden_state=lstm_hidden,
                    past_key_values=past_key_values
                )

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences.unsqueeze(1) + (pad_token_id * (1 - unfinished_sequences)).unsqueeze(1)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens.long()], dim=-1)
            cur_len = cur_len + self.ds_factor

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished = (next_tokens != eos_token_id).min(dim=-1)[0]
                unfinished_sequences = unfinished_sequences.mul(unfinished.long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break

        return input_ids
    
    