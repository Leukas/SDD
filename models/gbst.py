import math
from math import gcd
import functools
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def lcm(*numbers):
    return int(functools.reduce(lambda x, y: int((x * y) / gcd(x, y)), numbers, 1))

def masked_mean(tensor, mask, dim = -1):
    diff_len = len(tensor.shape) - len(mask.shape)
    mask = mask[(..., *((None,) * diff_len))] # gives mask the same number of dims as tensor
    tensor.masked_fill_(~mask, 0.)

    total_el = mask.sum(dim = dim).float()
    mean = tensor.sum(dim = dim) / total_el.clamp(min = 1.)
    mean.masked_fill_(total_el == 0, 0.) # isn't this redundant?
    return mean

def next_divisible_length(seqlen, multiple):
    return math.ceil(seqlen / multiple) * multiple

def pad_to_multiple(tensor, multiple, *, seq_dim, dim = -1, value = 0.):
    seqlen = tensor.shape[seq_dim]
    length = next_divisible_length(seqlen, multiple)
    if length == seqlen:
        return tensor
    remainder = length - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value = value)

# helper classes

class Pad(nn.Module):
    def __init__(self, padding, value = 0.):
        super().__init__()
        self.padding = padding
        self.value = value

    def forward(self, x):
        return F.pad(x, self.padding, value = self.value)

class DepthwiseConv1d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, kernel_size, groups = dim_in)
        self.proj_out = nn.Conv1d(dim_out, dim_out, 1)

    def forward(self, x):
        x = self.conv(x)
        return self.proj_out(x)


def create_sinusoidal_embeddings(n_pos, dim, out):
    import numpy as np
    out.requires_grad = False
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for pos in range(n_pos)
    ])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
# main class

class GBST(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        max_block_size = None,
        blocks = None,
        downsample_factor = 4,
        score_consensus_attn = True,
        causal = False,
        no_conv = False,
        lang_embs = False
    ):
        super().__init__()
        assert exists(max_block_size) ^ exists(blocks), 'either max_block_size or blocks are given on initialization'
        self.token_emb = nn.Embedding(num_tokens, dim)

        if exists(blocks):
            assert isinstance(blocks, tuple), 'blocks must be a tuple of block sizes'
            self.blocks = tuple(map(lambda el: el if isinstance(el, tuple) else (el, 0), blocks))
            assert all([(offset < block_size) for block_size, offset in self.blocks]), 'offset must be always smaller than the block size'

            max_block_size = max(list(map(lambda t: t[0], self.blocks)))
        else:
            self.blocks = tuple(map(lambda el: (el, 0), range(1, max_block_size + 1)))

        self.causal = causal
        self.no_conv = no_conv
        if causal or no_conv:
            self.pos = nn.Embedding(downsample_factor, dim)
            self.pos.requires_grad = False
            create_sinusoidal_embeddings(downsample_factor, dim, self.pos.weight)
        else:
            self.pos = nn.Sequential(
                Pad((0, 0, 0, max_block_size - 1)),
                Rearrange('b n d -> b d n'),
                DepthwiseConv1d(dim, dim, kernel_size = max_block_size),
                Rearrange('b d n -> b n d')
            )

        if lang_embs:
            self.lang_emb = nn.Embedding(2, dim)

        self.score_fn = nn.Sequential(
            nn.Linear(dim, 1),
            Rearrange('... () -> ...')
        )

        self.score_consensus_attn = score_consensus_attn

        assert downsample_factor <= max_block_size, 'final downsample factor should be less than the maximum block size'

        self.block_pad_multiple = lcm(*[block_size for block_size, _ in self.blocks])
        self.downsample_factor = downsample_factor

    def forward(self, x, mask = None, lang = None):
        b, n, block_mult, ds_factor, device = *x.shape, self.block_pad_multiple, self.downsample_factor, x.device
        m = next_divisible_length(n, ds_factor)

        # get character token embeddings

        x_emb = self.token_emb(x)

        # do a conv to generate the positions for the tokens
        if self.causal or self.no_conv:
            positions = repeat(torch.arange(self.downsample_factor).to(x.device), 'n -> b n m', b=b, m=m//self.downsample_factor)
            positions = rearrange(positions, 'b n m -> b (m n)')
            positions = positions[:, :x.size(1)]
            x_pos = self.pos(positions)
            x = x_emb + x_pos 
        else:
            x = self.pos(x_emb)

        if lang is not None and hasattr(self, 'lang_emb'):
            x = self.lang_emb(lang)

        # pad both sequence and mask to length visibile by all block sizes from 0 to max block size

        x = pad_to_multiple(x, block_mult, seq_dim = 1, dim = -2)

        if exists(mask):
            mask = pad_to_multiple(mask, block_mult, seq_dim = 1, dim = -1, value = False)

        # causal_mask = None
        # if self.causal:
        #     init_mask = repeat(torch.eye(n), 'i j -> b i j', b=b)
        #     padded_size = (b, x.size(1), x.size(1))
        #     causal_mask = torch.zeros(padded_size)
        #     causal_mask[:, :n, :n] = init_mask
        #     print(causal_mask)

        # compute representations for all blocks by mean pooling

        block_masks = []
        block_reprs = []

        for block_size, offset in self.blocks:
            if self.downsample_factor % block_size > 0 and self.causal:
                continue # HACK: for removing information leak
            # clone the input sequence as well as the mask, in order to pad for offsets

            block_x = x.clone()

            if exists(mask):
                block_mask = mask.clone()

            # pad for offsets, if needed

            need_padding = offset > 0

            if need_padding:
                left_offset, right_offset = (block_size - offset), offset
                block_x = F.pad(block_x, (0, 0, left_offset, right_offset), value = 0.)

                if exists(mask):
                    block_mask = F.pad(block_mask, (left_offset, right_offset), value = False)

            # group input sequence into blocks

            blocks = rearrange(block_x, 'b (n m) d -> b n m d', m = block_size)

            # either mean pool the blocks, or do a masked mean

            if exists(mask):
                mask_blocks = rearrange(block_mask, 'b (n m) -> b n m', m = block_size)
                block_repr = masked_mean(blocks, mask_blocks, dim = -2)
            else:
                block_repr = blocks.mean(dim = -2)

            # append the block representations, as well as the pooled block masks

            block_repr = repeat(block_repr, 'b n d -> b (n m) d', m = block_size)

            if need_padding:
                block_repr = block_repr[:, left_offset:-right_offset]

            block_reprs.append(block_repr)

            if exists(mask):
                mask_blocks = torch.any(mask_blocks, dim = -1)
                mask_blocks = repeat(mask_blocks, 'b n -> b (n m)', m = block_size)

                if need_padding:
                    mask_blocks = mask_blocks[:, left_offset:-right_offset]

                block_masks.append(mask_blocks)

        # stack all the block representations

        block_reprs = torch.stack(block_reprs, dim = 2)

        # calculate scores and softmax across the block size dimension

        scores = self.score_fn(block_reprs)

        if exists(mask):
            block_masks = torch.stack(block_masks, dim = 2)
            max_neg_value = -torch.finfo(scores.dtype).max
            scores = scores.masked_fill(~block_masks, max_neg_value)

        scores = scores.softmax(dim = 2)

        # do the cheap consensus attention, eq (5) in paper

        if self.score_consensus_attn:
            score_sim = einsum('b i d, b j d -> b i j', scores, scores)

            if exists(mask):
                cross_mask = rearrange(mask, 'b i -> b i ()') * rearrange(mask, 'b j -> b () j')
                max_neg_value = -torch.finfo(score_sim.dtype).max
                score_sim = score_sim.masked_fill(~cross_mask, max_neg_value)

            score_attn = score_sim.softmax(dim = -1)
            scores = einsum('b i j, b j m -> b i m', score_attn, scores)

        # multiply the block representations by the position-wise scores

        scores = rearrange(scores, 'b n m -> b n m ()')
        x = (block_reprs * scores).sum(dim = 2)

        # truncate to length divisible by downsample factor

        x = x[:, :m]

        if exists(mask):
            mask = mask[:, :m]

        # final mean pooling downsample

        x = rearrange(x, 'b (n m) d -> b n m d', m = ds_factor)

        if exists(mask):
            mask = rearrange(mask, 'b (n m) -> b n m', m = ds_factor)
            x = masked_mean(x, mask, dim = 2)
            mask = torch.any(mask, dim = -1)
        else:
            x = x.mean(dim = -2)

        return x, mask



if __name__ == "__main__":

    gbst = GBST(num_tokens=50, dim=2, max_block_size=4, score_consensus_attn=False, causal=True)
    gbst.token_emb.weight = torch.nn.Parameter(torch.arange(100).view(50,2).float())

    # print(gbst.score_fn[0].bias)
    gbst.score_fn[0].weight[0] = torch.nn.Parameter(torch.Tensor([0.5, 0.5]))
    gbst.score_fn[0].bias = torch.nn.Parameter(torch.Tensor([0]))
    # print(gbst.pos_conv[2])
    mask = torch.ones(10)
    mask[8:] = 0
    mask = mask.unsqueeze(0).bool()
    x = gbst(torch.arange(10).view(1,10), mask)# , mask=torch.eye(10).view(1, 10, 10))
    print(x[0].size())
    print(x[1])
