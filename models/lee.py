# lee.py
""" Modified code from: https://github.com/jlibovicky/char-nmt-two-step-decoder/blob/master/encoder.py """

import math
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import apply_chunking_to_forward

from typing import Tuple, List

T = torch.Tensor

class Highway(nn.Module):
    """Highway layer.

    https://arxiv.org/abs/1507.06228

    Adapted from:
    https://gist.github.com/dpressel/3b4780bafcef14377085544f44183353
    """
    def __init__(self, input_size: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Conv1d(input_size, input_size, kernel_size=1, stride=1)
        self.transform = nn.Conv1d(
            input_size, input_size, kernel_size=1, stride=1)
        self.transform.bias.data.fill_(-2.0) # type: ignore
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: T) -> T:
        proj_result = F.relu(self.proj(x))
        proj_gate = torch.sigmoid(self.transform(x))
        gated = (proj_gate * proj_result) + ((1 - proj_gate) * x)
        return gated


class TransformerFeedForward(nn.Module):
    """Feedforward sublayer from the Transformer."""
    def __init__(
            self, input_size: int,
            intermediate_size: int, dropout: float) -> None:
        super().__init__()

        self.sublayer = nn.Sequential(
            nn.Linear(input_size, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, input_size),
            nn.Dropout(dropout))

        self.norm = nn.LayerNorm(input_size)

    def forward(self, input_tensor: T) -> T:
        output = self.sublayer(input_tensor)
        return self.norm(output + input_tensor)

DEFAULT_FILTERS = [128, 256, 512, 512, 256]


class CharToPseudoWord(nn.Module):
    """Character-to-pseudoword encoder."""
    # pylint: disable=too-many-arguments
    def __init__(
            self, input_dim: int,
            # pylint: disable=dangerous-default-value
            conv_filters: List[int] = DEFAULT_FILTERS,
            # pylint: enable=dangerous-default-value
            intermediate_cnn_layers: int = 0,
            intermediate_dim: int = 512,
            highway_layers: int = 2,
            ff_layers: int = 2,
            max_pool_window: int = 5,
            dropout: float = 0.1,
            is_decoder: bool = False) -> None:
        super().__init__()

        self.is_decoder = is_decoder
        self.conv_count = len(conv_filters)
        self.max_pool_window = max_pool_window
        # DO NOT PAD IN DECODER, is handled in forward
        if conv_filters == [0]:
            self.convolutions = None
            self.conv_count = 0
            self.cnn_output_dim = input_dim
        else:
            # TODO maybe the correct padding for the decoder is just 2 * i
            self.convolutions = nn.ModuleList([
                nn.Conv1d(
                    input_dim, dim, kernel_size=2 * i + 1, stride=1,
                    padding=2 * i if is_decoder else i)
                for i, dim in enumerate(conv_filters)])
            self.cnn_output_dim = sum(conv_filters)

        self.after_cnn = nn.Sequential(
            nn.Conv1d(self.cnn_output_dim, intermediate_dim, 1),
            nn.Dropout(dropout),
            nn.ReLU())

        self.intermediate_cnns = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    intermediate_dim, 2 * intermediate_dim, 3, padding=1),
                nn.Dropout(dropout))
            for _ in range(intermediate_cnn_layers)])
        self.intermediate_cnn_norm = nn.ModuleList([
            nn.LayerNorm(intermediate_dim)
            for _ in range(intermediate_cnn_layers)])

        self.shrink = nn.MaxPool1d(
            max_pool_window, max_pool_window,
            padding=0, ceil_mode=True)

        self.highways = nn.Sequential(
            *(Highway(intermediate_dim, dropout)
              for _ in range(highway_layers)))

        self.after_highways = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(intermediate_dim))

        self.ff_layers = nn.Sequential(
            *(TransformerFeedForward(
                intermediate_dim, 2 * intermediate_dim, dropout)
              for _ in range(ff_layers)))

        self.final_mask_shrink = nn.MaxPool1d(
            max_pool_window, max_pool_window, padding=0, ceil_mode=True)
    # pylint: enable=too-many-arguments

    def forward(self, embedded_chars: T, mask: T) -> Tuple[T, T]:
        embedded_chars = embedded_chars.transpose(2, 1)
        conv_mask = mask.unsqueeze(1)

        if self.convolutions is not None:
            conv_outs = []
            for i, conv in enumerate(self.convolutions):
                conv_i_out = conv(embedded_chars * conv_mask)
                if self.is_decoder and i > 0:
                    conv_i_out = conv_i_out[:, :, :-2 * i]
                conv_outs.append(conv_i_out)

            convolved_char = torch.cat(conv_outs, dim=1)
        else:
            convolved_char = embedded_chars

        convolved_char = self.after_cnn(convolved_char)
        for cnn, norm in zip(self.intermediate_cnns,
                             self.intermediate_cnn_norm):
            conv_out = F.glu(cnn(convolved_char * conv_mask), dim=1)
            convolved_char = norm(
                (conv_out + convolved_char).transpose(2, 1)).transpose(2, 1)

        shrinked = self.shrink(convolved_char)
        output = self.highways(shrinked).transpose(2, 1)
        output = self.after_highways(output)
        output = self.ff_layers(output)
        shrinked_mask = self.final_mask_shrink(mask.unsqueeze(1)).squeeze(1)

        return output, shrinked_mask

class CharToWord(nn.Module):
    """Character-to-word encoder."""
    # pylint: disable=too-many-arguments
    def __init__(
            self, input_dim: int,
            # pylint: disable=dangerous-default-value
            conv_filters: List[int] = DEFAULT_FILTERS,
            # pylint: enable=dangerous-default-value
            intermediate_cnn_layers: int = 0,
            intermediate_dim: int = 512,
            highway_layers: int = 2,
            ff_layers: int = 2,
            dropout: float = 0.1,
            is_decoder: bool = False) -> None:
        super().__init__()

        self.is_decoder = is_decoder
        self.conv_count = len(conv_filters)
        # DO NOT PAD IN DECODER, is handled in forward
        if conv_filters == [0]:
            self.convolutions = None
            self.conv_count = 0
            self.cnn_output_dim = input_dim
        else:
            # TODO maybe the correct padding for the decoder is just 2 * i
            self.convolutions = nn.ModuleList([
                nn.Conv1d(
                    input_dim, dim, kernel_size=2 * i + 1, stride=1,
                    padding=2 * i if is_decoder else i)
                for i, dim in enumerate(conv_filters)])
            self.cnn_output_dim = sum(conv_filters)

        self.after_cnn = nn.Sequential(
            nn.Conv1d(self.cnn_output_dim, intermediate_dim, 1),
            nn.Dropout(dropout),
            nn.ReLU())

        self.intermediate_cnns = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    intermediate_dim, 2 * intermediate_dim, 3, padding=1),
                nn.Dropout(dropout))
            for _ in range(intermediate_cnn_layers)])
        self.intermediate_cnn_norm = nn.ModuleList([
            nn.LayerNorm(intermediate_dim)
            for _ in range(intermediate_cnn_layers)])

        self.highways = nn.Sequential(
            *(Highway(intermediate_dim, dropout)
              for _ in range(highway_layers)))

        self.after_highways = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(intermediate_dim))

        self.ff_layers = nn.Sequential(
            *(TransformerFeedForward(
                intermediate_dim, 2 * intermediate_dim, dropout)
              for _ in range(ff_layers)))


    @staticmethod
    def get_causal_block_mask(batch_size, orig_len, idxs, device):
        ds_len = idxs.size(-1)
        mask = torch.zeros(batch_size, ds_len, orig_len, device=device)
        idxs2 = torch.cat((torch.zeros(batch_size, 1, device=device), idxs[:, :-1]), dim=1)
        for i in range(batch_size):
            mask[i] = torch.arange(orig_len, device=device) < idxs[i][:, None] 
            mask[i] *= torch.arange(orig_len, device=device) >= idxs2[i][:, None]

        return mask # b s n

    @staticmethod
    def get_full_block_mask(batch_size, orig_len, idxs, device):
        char_lens = idxs[:, -1] # b
        mask = torch.arange(orig_len, device=device) < char_lens[:, None] # b n
        return mask # b n

    @staticmethod
    def masked_max(tensor, mask):
        # tensor.masked_fill_(~mask, 0.)
        # tensor.masked_fill_(~mask, 0.)
        tmul = tensor * mask
        mean = tmul.max(dim=-1)[0]
        return mean

    def forward(self, embedded_chars: T, mask: T, word_lens) -> Tuple[T, T]:
        shrinked_mask = word_lens != -100
        total_lens = shrinked_mask.sum(dim=1)
        batch_size = len(total_lens)
        orig_len = embedded_chars.size(1)


        embedded_chars = embedded_chars.transpose(2, 1)
        conv_mask = mask.unsqueeze(1)

        if self.convolutions is not None:
            conv_outs = []
            for i, conv in enumerate(self.convolutions):
                conv_i_out = conv(embedded_chars * conv_mask)
                if self.is_decoder and i > 0:
                    conv_i_out = conv_i_out[:, :, :-2 * i]
                conv_outs.append(conv_i_out)

            convolved_char = torch.cat(conv_outs, dim=1)
        else:
            convolved_char = embedded_chars

        convolved_char = self.after_cnn(convolved_char)
        for cnn, norm in zip(self.intermediate_cnns,
                             self.intermediate_cnn_norm):
            conv_out = F.glu(cnn(convolved_char * conv_mask), dim=1)
            convolved_char = norm(
                (conv_out + convolved_char).transpose(2, 1)).transpose(2, 1)

        word_lens_sum = torch.cumsum(word_lens.clamp(0), dim=1).long()
        causal_mask = CharToWord.get_causal_block_mask(batch_size, orig_len, word_lens_sum, convolved_char.device)

        convolved_char = convolved_char.unsqueeze(2) # b d 1 n
        causal_mask = causal_mask.unsqueeze(1) # b 1 s n

        # mask and downsample in chunks of batch_size//8 because doing it in one go is too memory intensive
        # NOTE: Could increase chunk size for faster processing in some cases
        # pad batch_size to multiple of 8
        mul_size = 8
        batch_size_mul = math.ceil(batch_size/mul_size) * mul_size
        if batch_size_mul != batch_size:
            pad_rest = batch_size_mul - batch_size
            padding = torch.zeros((pad_rest, *convolved_char.shape[1:]), dtype=convolved_char.dtype, device=convolved_char.device)
            convolved_char = torch.cat((convolved_char, padding))
            mask_padding = torch.zeros((pad_rest, *causal_mask.shape[1:]), dtype=causal_mask.dtype, device=causal_mask.device)
            causal_mask = torch.cat((causal_mask, mask_padding))
            shrinked = apply_chunking_to_forward(CharToWord.masked_max, batch_size_mul//mul_size, 0, convolved_char, causal_mask)[:batch_size]
        else:
            shrinked = apply_chunking_to_forward(CharToWord.masked_max, batch_size//mul_size, 0, convolved_char, causal_mask)[:batch_size]

        output = self.highways(shrinked).transpose(2, 1)
        output = self.after_highways(output)
        output = self.ff_layers(output)

        return output, shrinked_mask

