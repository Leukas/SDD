import torch
import torch.nn as nn
from einops import rearrange
from transformers.modeling_outputs import Seq2SeqLMOutput
from models.charformer_bert import BertWithGBSTEncDec

class TwoStep(BertWithGBSTEncDec):
    def __init__(self, pretrained_model, bert_config=None):
        super().__init__(pretrained_model, bert_config, True, 1)

    @classmethod
    def from_pretrained(cls, reload_folder, config=None):
        model = cls(None, config)
        state_dict_path = reload_folder + "/pytorch_model.bin"
        model.load_state_dict(torch.load(state_dict_path))
        return model

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        out = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
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
        out = self.bert.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            return_dict=True)
        return out.last_hidden_state, attention_mask


    def decode_lstm_generate(self, decoder_input_ids, decoder_attention_mask, encoder_hidden_states, encoder_attention_mask, lstm_hidden_state, past_key_values):
        assert encoder_hidden_states is not None

        out = self.bert.decoder(
            input_ids=decoder_input_ids[:, -1:], 
            attention_mask=decoder_attention_mask,
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